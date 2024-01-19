#if HAVE_CONFIG_H
#include <config.h>
#endif

#include <Python.h>

#include <memory>

#include <dune/python/pybind11/numpy.h>
#include <dune/python/pybind11/operators.h>
#include <dune/python/pybind11/pybind11.h>
#include <dune/python/pybind11/stl.h>

#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>

#include <duneuro/common/dense_matrix.hh>
#include <duneuro/common/points_on_sphere.hh>
#include <duneuro/io/dipole_reader.hh>
#include <duneuro/io/field_vector_reader.hh>
#include <duneuro/io/point_vtk_writer.hh>
#include <duneuro/io/projections_reader.hh>
#include <duneuro/driver/driver_factory.hh>
#include <duneuro/driver/driver_interface.hh>
#include <duneuro/py/dipole_statistics.hh>
#include <duneuro/py/parameter_tree.h>
#include <duneuro/tes/patch_set.hh>
#include <duneuro/tes/tdcs_driver_factory.hh>
#include <duneuro/tes/tdcs_driver_interface.hh>
#if HAVE_DUNE_UDG
#include <duneuro/udg/hexahedralization.hh>
#include <duneuro/udg/unfitted_statistics.hh>
#endif

namespace py = pybind11;

static inline void register_exceptions()
{
  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p)
        std::rethrow_exception(p);
    } catch (const Dune::Exception& e) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
    }
  });
}

struct ParameterTreeStorage : public duneuro::StorageInterface {
public:
  explicit ParameterTreeStorage(int verbose = 0)
   : verbose_(verbose)
  {
  }

  virtual void store(const std::string& name, const std::string& value)
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (verbose_ >= 1) {
      std::cout << name << " = " << value << "\n";
    }
    tree[name] = value;
  }

  virtual void storeMatrix(const std::string& name,
                           std::shared_ptr<duneuro::MatrixInterface<double>> matrix)
  {
  }

  virtual void storeMatrix(const std::string& name,
                           std::shared_ptr<duneuro::MatrixInterface<unsigned int>> matrix)
  {
  }

  Dune::ParameterTree tree;
  std::mutex mutex;
  int verbose_;
};

std::unique_ptr<duneuro::DenseMatrix<double>> toDenseMatrix(py::buffer buffer)
{
  /* Request a buffer descriptor from Python */
  py::buffer_info info = buffer.request();

  /* Some sanity checks ... */
  if (info.format != py::format_descriptor<double>::value)
    throw std::runtime_error("Incompatible format: expected a double array!");

  if (info.ndim != 2)
    throw std::runtime_error("Incompatible buffer dimension!");

  if (info.strides[1] / sizeof(double) != 1)
    throw std::runtime_error("Supporting only row major format");

  return std::make_unique<duneuro::DenseMatrix<double>>(info.shape[0], info.shape[1],
                                                              static_cast<double*>(info.ptr));
}

#if HAVE_DUNE_UDG
template <int dim>
static duneuro::UnfittedMEEGDriverData<dim> extractUnfittedDataFromMainDict(py::dict d)
{
  duneuro::UnfittedMEEGDriverData<dim> data;
  if (d.contains("domain") && d["domain"].contains("level_sets")) {
    auto domainDict = d["domain"].cast<py::dict>();
    auto list = domainDict["level_sets"].cast<py::list>();
    for (auto lvlst : list) {
      auto levelsetDict = domainDict[lvlst].cast<py::dict>();
      if (levelsetDict["type"].cast<std::string>() == "image") {
        if (levelsetDict.contains("data")) {
          auto buffer = levelsetDict["data"].cast<py::buffer>();
          py::buffer_info info = buffer.request();
          if (info.format != py::format_descriptor<double>::value) {
            DUNE_THROW(Dune::Exception, "only float level sets are supported. expected "
                                            << py::format_descriptor<double>::value << " got "
                                            << info.format);
          }
          // yasp grid uses col major ordering of the vertices in its index sets, which
          // are then used by dune-functions to determine the index
          if (info.strides[0] != sizeof(double)) {
            DUNE_THROW(Dune::Exception, "only F-style (col major) ordering is supported");
          }
          std::array<unsigned int, dim> elementsInDim;
          std::copy(info.shape.begin(), info.shape.end(), elementsInDim.begin());
          duneuro::SimpleStructuredGrid<dim> grid(elementsInDim);
          double* ptr = reinterpret_cast<double*>(info.ptr);
          auto imagedata = std::make_shared<std::vector<double>>(ptr, ptr + info.size);
          data.levelSetData.images.push_back(
              std::make_shared<duneuro::Image<double, dim>>(imagedata, grid));
          levelsetDict["image_index"] = py::int_(data.levelSetData.images.size() - 1);
        }
      }
    }
  }
  return data;
}
#endif

// create basic binding for a dune field vector
template <class T, int dim>
void register_field_vector(py::module& m)
{
  using FieldVector = Dune::FieldVector<T, dim>;
  std::stringstream docstr;
  docstr << "a " << dim << "-dimensional vector";
  py::class_<FieldVector>(m, (std::string("FieldVector") + std::to_string(dim) + "D").c_str(),
                          docstr.str().c_str(), py::buffer_protocol())
      .def_buffer([](FieldVector& m) -> py::buffer_info {
        return py::buffer_info(
            &m[0], /* Pointer to buffer */
            sizeof(T), /* Size of one scalar */
            py::format_descriptor<T>::value, /* Python struct-style format descriptor */
            1, /* Number of dimensions */
            {dim}, /* Buffer dimensions */
            {sizeof(T)});
      })
      .def(py::init<T>())
      .def(py::init(
        [] (py::array_t<T> array) {
          if (array.size() != dim) {
            DUNE_THROW(Dune::Exception, "array has to have size " << dim << " but got" << array.size());
          }
          if (array.ndim() != 1) {
            DUNE_THROW(Dune::Exception, "array has to have dim " << 1 << " but got " << array.ndim());
          }
          FieldVector vector(0.0);
          const T* data_ptr = array.data();
          std::copy(data_ptr, data_ptr + dim, vector.begin());
          return vector;
        }), // end definition of lambda
        "create a vector from any python buffer, such as a numpy array"
      ) // end definition of constructor
      .def(py::self += py::self)
      .def(py::self -= py::self)
      .def(py::self += T())
      .def(py::self -= T())
      .def(py::self *= T())
      .def(py::self /= T())
      .def("__len__", &FieldVector::size)
      .def("__getitem__",
           [](const FieldVector& instance, std::size_t i) {
             assert(i < dim);
             return instance[i];
           })
      .def("__str__", [](const FieldVector& fv) {
        std::stringstream sstr;
        sstr << "FieldVector" << dim << "d[" << fv << "]";
        return sstr.str();
      });
}

// create bindings for a dipole
template <class T, int dim>
void register_dipole(py::module& m)
{
  using Dipole = duneuro::Dipole<T, dim>;
  using FieldVector = Dune::FieldVector<T, dim>;
  std::stringstream classname;
  classname << "Dipole" << dim << "d";
  py::class_<Dipole>(
      m, classname.str().c_str(),
      "a class representing a mathematical dipole consisting of a position and moment")
      .def(py::init<FieldVector, FieldVector>(), "create a dipole from its position and moment",
           py::arg("position"), py::arg("moment"))
      .def(py::init(
           [](py::array_t<T> pos, py::array_t<T> mom) {
             if (pos.size() != dim || pos.ndim() != 1)
               DUNE_THROW(Dune::Exception, "position has to have " << dim << " entries");
             if (mom.size() != dim || pos.ndim() != 1)
               DUNE_THROW(Dune::Exception, "moment has to have " << dim << " entries");
             FieldVector vpos, vmom;
             std::copy(pos.data(), pos.data() + dim, vpos.begin());
             std::copy(mom.data(), mom.data() + dim, vmom.begin());
             return Dipole(vpos, vmom);
           }), // end definition of lambda
           "create a dipole from its position and moment", py::arg("position"), py::arg("moment")
      ) // end definition of constructor
      .def(py::init(
           [](py::array_t<T> pos_and_mom) {
             if(pos_and_mom.size() != 2 * dim || pos_and_mom.ndim() != 1) {
               DUNE_THROW(Dune::Exception, "combined buffer has to be 1-dimensional with" << 2 * dim << " entries");
             }

             FieldVector vpos, vmom;
             const T* data_ptr = pos_and_mom.data();
             std::copy(data_ptr, data_ptr + dim, vpos.begin());
             std::copy(data_ptr + dim, data_ptr + 2*dim, vmom.begin());
             return Dipole(vpos, vmom);
           }), // end definition of lambda
           "create a dipole from an array or list containing both its position and moment, where we assume the position is given first",
           py::arg("combined position and moment")
      ) // end definition of constructor
      .def("position", &Dipole::position, "return the dipole position",
           py::return_value_policy::reference_internal)
      .def("moment", &Dipole::moment, "return the dipole moment",
           py::return_value_policy::reference_internal)
      .def("__str__", [](const Dipole& dip) {
        std::stringstream sstr;
        sstr << "Dipole[position: [" << dip.position() << "] moment: [" << dip.moment() << "]]";
        return sstr.str();
      });
}

template <class T, int dim>
void register_read_dipoles(py::module& m)
{
  std::stringstream str;
  str << "read_dipoles_" << dim << "d";
  m.def(str.str().c_str(),
        [](const std::string& filename) { return duneuro::DipoleReader<T, dim>::read(filename); },
        R"pydoc(
read dipoles from a file

Each line of the file contains the position and moment of a single dipole as whitespace separated values,
e.g. for 3d::

  127 127 127 1 0 0
  128 128 127 0 1 0

for a file with two dipoles.
)pydoc");
}

template <class T, int dim>
void register_field_vector_reader(py::module& m)
{
  auto name = "read_field_vectors_" + std::to_string(dim) + "d";
  m.def(name.c_str(),
        [](const std::string& filename) {
          return duneuro::FieldVectorReader<T, dim>::read(filename);
        },
        R"pydoc(
read field vectors from a file

Each line of the file contains the entries of a single field vector, e.g. for 3d::

  127 127 127
  1 0 1

for a file with two field vectors.
  )pydoc");
}

template <class T, int dim>
void register_projections_reader(py::module& m)
{
  auto name = "read_projections_" + std::to_string(dim) + "d";
  m.def(name.c_str(),
        [](const std::string& filename) {
          return duneuro::ProjectionsReader<T, dim>::read(filename);
        },
        R"pydoc(
read projections from a field

Each line of the file belongs to the projections of a single coil. The projections are field
vectors and the values should be separated by white spaces, e.g. for 3d::

  1 0 0 0 1 0 0 0 1
  1 0 0 0 1 0 0 0 1

for a projections file with the 3 cartesian directions for two coils.
  )pydoc");
}

static inline void register_function(py::module& m)
{
  py::class_<duneuro::Function>(m, "FunctionWrapper", "a class representing a domain function");
}

#if HAVE_DUNE_UDG
template <int dim>
void register_hexahedralize(py::module& py)
{
  auto name = "hexahedralize_" + std::to_string(dim) + "d";
  py.def(name.c_str(), [](py::dict d) {
    auto data = extractUnfittedDataFromMainDict<dim>(d);
    return duneuro::hexahedralize(data, duneuro::toParameterTree(d));
  });
}

template <int dim>
void register_unfitted_statistics(py::module& m)
{
  using US = duneuro::UnfittedStatistics<dim>;
  auto name = "UnfittedStatistics" + std::to_string(dim) + "d";
  py::class_<US>(m, name.c_str(), "statistics of an unfitted discretization")
      .def(py::init(
           [](py::dict d) {
             auto ud = extractUnfittedDataFromMainDict<dim>(d);
             return US(ud, duneuro::toParameterTree(d));
           }))
      .def("interfaceValues", &US::interfaceValues, "evaluate the interfaces at a given position.");
}

#endif

class VolumeVTKWriter {
public:
  explicit VolumeVTKWriter(std::unique_ptr<duneuro::VolumeConductorVTKWriterInterface> volumeWriterPtr)
  : volumeWriterPtr_(std::move(volumeWriterPtr))
  {
  }
  
  void addVertexData(const duneuro::Function& function, const std::string& name)
  {
    volumeWriterPtr_->addVertexData(function, name);
  }
  
  void addVertexDataGradient(const duneuro::Function& function, const std::string& name)
  {
    volumeWriterPtr_->addVertexDataGradient(function, name);
  }
  
  void addCellData(const duneuro::Function& function, const std::string& name)
  {
    volumeWriterPtr_->addCellData(function, name);
  }
  
  void addCellDataGradient(const duneuro::Function& function, const std::string& name)
  {
    volumeWriterPtr_->addCellDataGradient(function, name);
  }
  
  void write(py::dict d) {
    volumeWriterPtr_->write(duneuro::toParameterTree(d));
  }
  
private:
  std::unique_ptr<duneuro::VolumeConductorVTKWriterInterface> volumeWriterPtr_;
};

void register_volume_vtk_writer(py::module& m)
{
  std::string name = "VolumeVTKWriter";
  py::class_<VolumeVTKWriter>(m, name.c_str(), "write a volume conductor and associated grid functions in the VTK format")
    .def("addVertexData", &VolumeVTKWriter::addVertexData, "evaluate the given function on each grid vertex")
    .def("addVertexData", &VolumeVTKWriter::addVertexDataGradient, "evaluate the gradient of the given function on each grid vertex")
    .def("addCellData", &VolumeVTKWriter::addCellData, "evaluate the given function at the center of each cell")
    .def("addCellDataGradient", &VolumeVTKWriter::addCellDataGradient, "evaluate the gradient of the given function at the center of each cell")
    .def("write", &VolumeVTKWriter::write, "write output");
}

template <int dim>
class PyMEEGDriverInterface
{
public:
  using Interface = duneuro::DriverInterface<dim>;
  explicit PyMEEGDriverInterface(py::dict d)
  {
    duneuro::MEEGDriverData<dim> data;
#if HAVE_DUNE_UDG
    data.unfittedData = extractUnfittedDataFromMainDict<dim>(d);
#endif
    duneuro::extractFittedDataFromMainDict(d, data.fittedData);
    driver_ = duneuro::DriverFactory<dim>::make_driver(duneuro::toParameterTree(d), data);
  }

  std::unique_ptr<duneuro::Function> makeDomainFunction() const
  {
    return driver_->makeDomainFunction();
  }

  py::dict solveEEGForward(const typename Interface::DipoleType& dipole,
                           duneuro::Function& solution, py::dict config)
  {
    int verbose = config.contains("solver") && config["solver"].contains("verbose") ? py::int_(config["solver"]["verbose"]).cast<int>() : 0;
    auto storage = std::make_shared<ParameterTreeStorage>(verbose);
    driver_->solveEEGForward(dipole, solution, duneuro::toParameterTree(config),
                             duneuro::DataTree(storage));
    return duneuro::toPyDict(storage->tree);
  }

  std::pair<std::vector<double>, py::dict> solveMEGForward(const duneuro::Function& eegSolution,
                                                           py::dict config)
  {
    int verbose = config.contains("solver") && config["solver"].contains("verbose") ? py::int_(config["solver"]["verbose"]).cast<int>() : 0;
    auto storage = std::make_shared<ParameterTreeStorage>(verbose);
    auto result = driver_->solveMEGForward(eegSolution, duneuro::toParameterTree(config),
                                           duneuro::DataTree(storage));
    return {result, duneuro::toPyDict(storage->tree)};
  }

  VolumeVTKWriter volumeConductorVTKWriter(py::dict config)
  {
    return VolumeVTKWriter(driver_->volumeConductorVTKWriter(duneuro::toParameterTree(config)));
  }

  void setElectrodes(const std::vector<typename Interface::CoordinateType>& electrodes,
                     py::dict config)
  {
    driver_->setElectrodes(electrodes, duneuro::toParameterTree(config));
  }

  std::vector<typename Interface::CoordinateType> getProjectedElectrodes()
  {
    return driver_->getProjectedElectrodes();
  }

  std::vector<double> evaluateAtElectrodes(const duneuro::Function& solution) const
  {
    return driver_->evaluateAtElectrodes(solution);
  }

  void setCoilsAndProjections(
      const std::vector<typename Interface::CoordinateType>& coils,
      const std::vector<std::vector<typename Interface::CoordinateType>>& projections)
  {
    driver_->setCoilsAndProjections(coils, projections);
  }

  std::pair<duneuro::DenseMatrix<double>*, py::dict> computeEEGTransferMatrix(py::dict config)
  {
    int verbose = config.contains("solver") && config["solver"].contains("verbose") ? py::int_(config["solver"]["verbose"]).cast<int>() : 0;
    auto storage = std::make_shared<ParameterTreeStorage>(verbose);
    std::unique_ptr<duneuro::DenseMatrix<double>> result = driver_->computeEEGTransferMatrix(
        duneuro::toParameterTree(config), duneuro::DataTree(storage));
    return {result.release(), duneuro::toPyDict(storage->tree)};
  }

  std::pair<duneuro::DenseMatrix<double>*, py::dict> computeMEGTransferMatrix(py::dict config)
  {
    int verbose = config.contains("solver") && config["solver"].contains("verbose") ? py::int_(config["solver"]["verbose"]).cast<int>() : 0;
    auto storage = std::make_shared<ParameterTreeStorage>(verbose);
    std::unique_ptr<duneuro::DenseMatrix<double>> result = driver_->computeMEGTransferMatrix(
        duneuro::toParameterTree(config), duneuro::DataTree(storage));
    return {result.release(), duneuro::toPyDict(storage->tree)};
  }

  std::pair<std::vector<std::vector<double>>, py::dict>
  applyEEGTransfer(py::buffer buffer, const std::vector<typename Interface::DipoleType>& dipoles,
                   py::dict config)
  {
    int verbose = config.contains("solver") && config["solver"].contains("verbose") ? py::int_(config["solver"]["verbose"]).cast<int>() : 0;
    auto transferMatrix = toDenseMatrix(buffer);
    auto storage = std::make_shared<ParameterTreeStorage>(verbose);
    auto result = driver_->applyEEGTransfer(
        *transferMatrix, dipoles, duneuro::toParameterTree(config), duneuro::DataTree(storage));
    return {result, duneuro::toPyDict(storage->tree)};
  }

  std::pair<std::vector<std::vector<double>>, py::dict>
  applyMEGTransfer(py::buffer buffer, const std::vector<typename Interface::DipoleType>& dipoles,
                   py::dict config)
  {
    int verbose = config.contains("solver") && config["solver"].contains("verbose") ? py::int_(config["solver"]["verbose"]).cast<int>() : 0;
    auto transferMatrix = toDenseMatrix(buffer);
    auto storage = std::make_shared<ParameterTreeStorage>(verbose);
    auto result = driver_->applyMEGTransfer(
        *transferMatrix, dipoles, duneuro::toParameterTree(config), duneuro::DataTree(storage));
    return {result, duneuro::toPyDict(storage->tree)};
  }

  std::vector<std::vector<double>> computeMEGPrimaryField(const std::vector<typename Interface::DipoleType>& dipoles, py::dict config)
  {
    return driver_->computeMEGPrimaryField(dipoles, duneuro::toParameterTree(config));
  }

  py::dict statistics()
  {
    auto storage = std::make_shared<ParameterTreeStorage>();
    driver_->statistics(duneuro::DataTree(storage));
    return duneuro::toPyDict(storage->tree);
  }

  void print_citations()
  {
     driver_->print_citations();
  }

private:
  std::unique_ptr<Interface> driver_;
  Dune::ParameterTree tree_;
};

template <int dim>
static inline void register_meeg_driver_interface(py::module& m)
{
  using Interface = PyMEEGDriverInterface<dim>;
  std::stringstream classname;
  classname << "MEEGDriver" << dim << "d";
  py::class_<Interface>(m, classname.str().c_str())
      .def(py::init<py::dict>(), R"pydoc(
create a new MEEGDriver

the real driver to be used can be configured using the dictionary.

:param config: dictionary to configure the driver

  The main driver type can be selected with the ``type`` attribute:

  .. code-block:: ini

    type = udg | fitted

  **Fitted MEEGDrivers** (i.e. ``type == fitted``)

  The *fitted* refers to the mesh, which is fitted to the model geometry. On this fitted mesh,
  different solvers can be selected.

  .. code-block:: ini

    solver_type = cg | dg
    element_type = tetrahedron | hexahedron

  For a fitted driver, the mesh can either be read from file, or provided directly from python.
  To read the mesh from a file, set

  .. code-block:: ini

    volume_conductor.tensors.filename = string
    volume_conductor.grid.filename = string

  To provide the mesh directly, set

  .. code-block:: ini

    volume_conductor.tensors.labels = list<int>
    volume_conductor.tensors.conductivities = list<double>
    volume_conductor.grid.nodes = list<vector<double>>
    volume_conductor.grid.elements = list<list<int>>

  If both variants are present, the mesh from the python datastructures will be preferred.

  For the discontinuous Galerkin driver (i.e. ``solver_type == dg``), further options are available

  .. code-block:: ini

    solver.edge_norm_type = cell | face | houston
    solver.penalty = double
    solver.weights = bool
    solver.scheme = sipg | nipg

  The edge norm type describes how the element diameter :math:`h` is calculated in the DG scheme.
  On an intersection :math:`\gamma_{ef}` between elements :math:`e` and :math:`f`, the different
  edge norm types are defined as

  *cell*
    :math:`h = \left(\frac{2|e||f|}{|e|+|f|}\right)^{\frac{1}{d}}`

  *face*
    :math:`h = |\gamma_{ef}|^{\frac{1}{d-1}}`

  *houston*
    :math:`h = \frac{\min(|e|,|f|)}{|\gamma_{ef}|}`

  Where :math:`|e|` and :math:`|f|` denote the :math:`d`-dimensional measures of elements :math:`e`
  and :math:`f` respectively and :math:`|\gamma_{ef}|` denotes the :math:`(d-1)`-dimension measure
  of the face :math:`\gamma_{ef}`.

  **UDG MEEGDriver** (i.e. ``type == udg``)

  For the UDG driver, the mesh does not resolve the model geometry.

  .. code-block:: ini

    compartments = 0 < int < 7
    volume_conductor.grid.cells = vector<int>
    volume_conductor.grid.lower_left= vector<double>
    volume_conductor.grid.upper_right= vector<double>
    volume_conductor.grid.refinements= 0 <= int
    solver.conductivities = vector<double>
    solver.edge_norm_type = cell | face | houston
    solver.penalty = double
    solver.weights = bool
    solver.scheme = sipg | nipg

  The model geometry is given through level sets functions.

  .. code-block:: ini

    domain.domains = vector<string>
    domain.<domain.domains>.positions = vector<string>
    domain.level_sets = vector<string>
    domain.<domain.level_sets>.type = sphere | image | cylinder

  the options for a spherical level set (i.e. ``domain.<domain.level_sets>.type == sphere``) are

  .. code-block:: ini

    domain.<domain.level_sets>.radius = double
    domain.<domain.level_sets>.center = vector<double>

  the options for a level set taken from an image (i.e. ``domain.<domain.level_sets>.type == image``) are

  .. code-block:: ini

    domain.<domain.level_sets>.filename = string

  the options for a cylindrical level set (i.e. ``domain.<domain.level_sets>.type == cylinder``) are

  .. code-block:: ini

    domain.<domain.level_sets>.radius = double
    domain.<domain.level_sets>.center = vector<double>
    domain.<domain.level_sets>.direction = vector<double>
    domain.<domain.level_sets>.length = double
      )pydoc",
           py::arg("config"))
      .def("makeDomainFunction", &Interface::makeDomainFunction, "create a domain function")
      .def("solveEEGForward", &Interface::solveEEGForward,
           R"pydoc(
solve the eeg forward problem and store the result in the given function

:param config: dictionary to configure the solution process

  .. code-block:: ini

    post_process = bool
    subtract_mean = bool

  The solution of the linear system can be configured by settings options in the `solver` sub tree:

  .. code-block:: ini

    solver.reduction = double

  The type of source model can be chosen by setting the `source_model.type` entry.

  .. code-block:: ini

    source_model.type = partial_integration | subtraction | venant

  Depending on the type of the driver, not all source models might be available.
           )pydoc")
      .def("solveMEGForward", &Interface::solveMEGForward
           /* , */
           /* "solve the meg forward problem and return the solution" */)
      .def("volumeConductorVTKWriter", &Interface::volumeConductorVTKWriter, "return a VTK writer for this volume conductor")
      .def("setElectrodes", &Interface::setElectrodes,
           "set the electrodes. subsequent calls to evaluateAtElectrodes will use these "
           "electrodes.",
           py::arg("electrodes"), py::arg("config"))
      .def("getProjectedElectrodes", &Interface::getProjectedElectrodes,
           "return the projected electrodes in global coordinates")
      .def("setCoilsAndProjections", &Interface::setCoilsAndProjections,
           "set coils and projections for meg. subsequent calls to solveMEGForward will use "
           "these "
           "coils and projections",
           py::arg("coils"), py::arg("projections"))
      .def("evaluateAtElectrodes", &Interface::evaluateAtElectrodes,
           "evaluate the given function at the set electrodes")
      .def("computeEEGTransferMatrix", &Interface::computeEEGTransferMatrix,
           "compute the eeg transfer matrix using the set electrodes", py::arg("config"))
      .def("computeMEGTransferMatrix", &Interface::computeMEGTransferMatrix,
           "compute the meg transfer matrix using the set coils and projections", py::arg("config"))
      .def("applyEEGTransfer", &Interface::applyEEGTransfer, "apply the eeg transfer matrix",
           py::arg("matrix"), py::arg("dipoles"), py::arg("config"))
      .def("applyMEGTransfer", &Interface::applyMEGTransfer, "apply the meg transfer matrix",
           py::arg("matrix"), py::arg("dipoles"), py::arg("config"))
      .def("computeMEGPrimaryField", &Interface::computeMEGPrimaryField, "compute the primary B field for the given dipoles", py::arg("dipoles"), py::arg("config"))
      .def("statistics", &Interface::statistics, "compute driver statistics")
      .def("print_citations", &Interface::print_citations, "list relevant publications");
}

template <class T>
void register_dense_matrix(py::module& m)
{
  using Matrix = duneuro::DenseMatrix<T>;
  py::class_<Matrix>(m, "Matrix", py::buffer_protocol()).def_buffer([](Matrix& m) -> py::buffer_info {
    return py::buffer_info(
        m.data(), /* Pointer to buffer */
        sizeof(T), /* Size of one scalar */
        py::format_descriptor<T>::value, /* Python struct-style format descriptor */
        2, /* Number of dimensions */
        {m.rows(), m.cols()}, /* Buffer dimensions */
        {sizeof(T) * m.cols(), /* Strides (in bytes) for each index */
         sizeof(T)});
  });
}

template <int dim>
static inline void register_points_on_sphere(py::module& m)
{
  std::stringstream str;
  str << "generate_points_on_sphere_" << dim << "d";
  m.def(str.str().c_str(),
        [](py::dict d) {
          auto ptree = duneuro::toParameterTree(d);
          return duneuro::generate_points_on_sphere<double, dim>(ptree);
        },
        R"pydoc(
generate approximately uniformly distributed points on a sphere

:param config: configuration for point generation
)pydoc",
        py::arg("config"));
}

template <class ctype, int dim>
static inline void register_point_vtk_writer(py::module& m)
{
  using Writer = duneuro::PointVTKWriter<ctype, dim>;
  std::stringstream classname;
  classname << "PointVTKWriter" << dim << "d";
  py::class_<Writer>(m, classname.str().c_str(), "write points to vtk")
      .def(py::init<const std::vector<Dune::FieldVector<ctype, dim>>&, bool>())
      .def(py::init<const duneuro::Dipole<ctype, dim>&>())
      .def("addVectorData", &Writer::addVectorData, "add vector data to the given points.")
      .def("addScalarData", &Writer::addScalarData, "add scalar data to the given points.")
      .def("write",
           [](const Writer& instance, const std::string& filename) { instance.write(filename); },
           "write the data to vtk");
}

template <class T, int dim>
static inline void register_patch_set(py::module& m)
{
  using PS = duneuro::PatchSet<T, dim>;
  std::stringstream name;
  name << "PatchSet" << dim << "d";
  py::class_<PS>(m, name.str().c_str()).def(py::init([](py::dict d) {
    return PS(duneuro::toParameterTree(d));
  }));
}

template <int dim>
class PyTDCSDriverInterface
{
public:
  using Interface = duneuro::TDCSDriverInterface<dim>;
  explicit PyTDCSDriverInterface(const duneuro::PatchSet<double, dim>& patchSet, py::dict d)
  {
    duneuro::TDCSDriverData<dim> data;
#if HAVE_DUNE_UDG
    data.udgData = extractUnfittedDataFromMainDict<dim>(d);
#endif
    duneuro::extractFittedDataFromMainDict(d, data.fittedData);
    driver_ = duneuro::TDCSDriverFactory<dim>::make_tdcs_driver(patchSet,
                                                                duneuro::toParameterTree(d), data);
  }

  std::unique_ptr<duneuro::Function> makeDomainFunction() const
  {
    return driver_->makeDomainFunction();
  }

  VolumeVTKWriter volumeConductorVTKWriter(py::dict config)
  {
    return VolumeVTKWriter(driver_->volumeConductorVTKWriter(duneuro::toParameterTree(config)));
  }

  py::dict solveTDCSForward(duneuro::Function& solution, py::dict config) const
  {
    int verbose = config.contains("solver") && config["solver"].contains("verbose") ? py::int_(config["solver"]["verbose"]).cast<int>() : 0;
    auto storage = std::make_shared<ParameterTreeStorage>(verbose);
    driver_->solveTDCSForward(solution, duneuro::toParameterTree(config),
                              duneuro::DataTree(storage));
    return duneuro::toPyDict(storage->tree);
  }

private:
  std::unique_ptr<Interface> driver_;
  Dune::ParameterTree tree_;
};

template <int dim>
static inline void register_tdcs_driver_interface(py::module& m)
{
  using Interface = PyTDCSDriverInterface<dim>;
  std::stringstream classname;
  classname << "TDCSDriver" << dim << "d";
  py::class_<Interface>(m, classname.str().c_str())
      .def(py::init<duneuro::PatchSet<double, dim>, py::dict>())
      .def("makeDomainFunction", &Interface::makeDomainFunction, "create a domain function")
      .def("volumeConductorVTKWriter", &Interface::volumeConductorVTKWriter, "return a VTK writer for this volume conductor")
      .def("solveTDCSForward", &Interface::solveTDCSForward);
}

PYBIND11_MODULE(duneuropy, m)
{
	m.doc() = "duneuropy library";

  register_exceptions();

  register_function(m);

  register_volume_vtk_writer(m);

  register_dense_matrix<double>(m);

  register_field_vector<double, 2>(m);
  register_dipole<double, 2>(m);
  register_read_dipoles<double, 2>(m);
  register_field_vector_reader<double, 2>(m);
  register_projections_reader<double, 2>(m);
  register_meeg_driver_interface<2>(m);
  register_points_on_sphere<2>(m);
  register_point_vtk_writer<double, 2>(m);
  register_patch_set<double, 2>(m);
  register_tdcs_driver_interface<2>(m);
#if HAVE_DUNE_UDG
  register_hexahedralize<2>(m);
  register_unfitted_statistics<2>(m);
#endif
  duneuro::register_dipole_statistics<2>(m);

  register_field_vector<double, 3>(m);
  register_dipole<double, 3>(m);
  register_read_dipoles<double, 3>(m);
  register_field_vector_reader<double, 3>(m);
  register_projections_reader<double, 3>(m);
  register_meeg_driver_interface<3>(m);
  register_points_on_sphere<3>(m);
  register_point_vtk_writer<double, 3>(m);
  register_patch_set<double, 3>(m);
  register_tdcs_driver_interface<3>(m);
#if HAVE_DUNE_UDG
  register_hexahedralize<3>(m);
  register_unfitted_statistics<3>(m);
#endif
  duneuro::register_dipole_statistics<3>(m);
}
