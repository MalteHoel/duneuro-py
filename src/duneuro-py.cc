#if HAVE_CONFIG_H
#include <config.h>
#endif

#include <Python.h>

#include <memory>

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>
#include <dune/common/std/memory.hh>

#include <duneuro/common/dense_matrix.hh>
#include <duneuro/common/points_on_sphere.hh>
#include <duneuro/eeg/eeg_analytic_solution.hh>
#include <duneuro/io/dipole_reader.hh>
#include <duneuro/io/field_vector_reader.hh>
#include <duneuro/io/point_vtk_writer.hh>
#include <duneuro/io/projections_reader.hh>
#include <duneuro/meeg/meeg_driver_factory.hh>
#include <duneuro/meeg/meeg_driver_interface.hh>
#include <duneuro/tes/patch_set.hh>
#include <duneuro/tes/tdcs_driver_factory.hh>
#include <duneuro/tes/tdcs_driver_interface.hh>

namespace py = pybind11;

static inline void register_exceptions()
{
  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p)
        std::rethrow_exception(p);
    } catch (const Dune::Exception& e) {
      PyErr_SetString(PyExc_RuntimeError, e.what().c_str());
    }
  });
}

struct ParameterTreeStorage : public duneuro::StorageInterface {
  virtual void store(const std::string& name, const std::string& value)
  {
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
};

py::dict toPyDict(const Dune::ParameterTree& tree)
{
  py::dict out;
  for (const auto& key : tree.getValueKeys()) {
    out[key.c_str()] = py::str(tree[key]);
  }
  for (const auto& subkey : tree.getSubKeys()) {
    out[subkey.c_str()] = toPyDict(tree.sub(subkey));
  }
  return out;
}

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

  return Dune::Std::make_unique<duneuro::DenseMatrix<double>>(info.shape[0], info.shape[1],
                                                              static_cast<double*>(info.ptr));
}

static inline std::string py_to_string(py::handle handle)
{
  std::string type = handle.get_type().attr("__name__").cast<std::string>();
  if (type == "str")
    return handle.cast<std::string>();
  else if (type == "int")
    return std::to_string(handle.cast<int>());
  else if (type == "float") {
    std::stringstream sstr;
    sstr << handle.cast<double>();
    return sstr.str();
  } else if (type == "bool")
    return std::to_string(handle.cast<bool>());
  else if ((type == "list") || (type == "tuple")) {
    std::stringstream str;
    unsigned int i = 0;
    for (auto it = handle.begin(); it != handle.end(); ++it, ++i) {
      if (i > 0)
        str << " ";
      str << py_to_string(*it);
    }
    return str.str();
  }
  DUNE_THROW(Dune::Exception, "type \"" << type << "\" not supported");
}

static inline std::map<std::string, std::string> dictToStringMap(py::dict dict)
{
  std::map<std::string, std::string> map;
  for (const auto& item : dict) {
    std::string type = item.second.get_type().attr("__name__").cast<std::string>();
    std::string key = py_to_string(item.first);
    if (type == "dict") {
      auto sub = dictToStringMap(item.second.cast<py::dict>());
      for (const auto& k : sub) {
        map[key + "." + k.first] = k.second;
      }
    } else {
      try {
        map[key] = py_to_string(item.second);
      } catch (Dune::Exception& ex) {
        // ignore entry
      }
    }
  }
  return map;
}

// translate a python dict to a Dune::ParameterTree. The dict can be nested and the sub dicts will
// be translated recursively.
static inline Dune::ParameterTree dictToParameterTree(py::dict dict)
{
  Dune::ParameterTree tree;
  auto map = dictToStringMap(dict);
  for (const auto& k : map) {
    tree[k.first] = k.second;
  }
  return tree;
}

template <int dim>
static void extractFittedDataFromMainDict(py::dict d, duneuro::FittedDriverData<dim>& data)
{
  if (d.contains("volume_conductor")) {
    auto volume_conductor_dict = d["volume_conductor"].cast<py::dict>();
    if (volume_conductor_dict.contains("grid") && volume_conductor_dict.contains("tensors")) {
      auto grid_dict = volume_conductor_dict["grid"].cast<py::dict>();
      auto tensor_dict = volume_conductor_dict["tensors"].cast<py::dict>();
      if (grid_dict.contains("nodes") && grid_dict.contains("elements")
          && tensor_dict.contains("labels") && tensor_dict.contains("conductivities")) {
        std::cout << "casting nodes" << std::endl;
        for (const auto& n : grid_dict["nodes"]) {
          auto arr = n.cast<std::vector<double>>();
          if (arr.size() != dim)
            DUNE_THROW(Dune::Exception, "each node has to have " << dim << " entries");
          typename duneuro::FittedDriverData<dim>::Coordinate p;
          std::copy(arr.begin(), arr.end(), p.begin());
          data.nodes.push_back(p);
        }
        for (const auto& e : grid_dict["elements"]) {
          data.elements.push_back(e.cast<std::vector<unsigned int>>());
          for (const auto& ei : data.elements.back()) {
            if (ei >= data.nodes.size()) {
              DUNE_THROW(Dune::Exception, "node index " << ei << " out of bounds ("
                                                        << data.nodes.size() << ")");
            }
          }
        }
        data.labels = tensor_dict["labels"].cast<std::vector<int>>();
        data.conductivities = tensor_dict["conductivities"].cast<std::vector<double>>();
        if (data.labels.size() != data.elements.size()) {
          DUNE_THROW(Dune::Exception, "labels and elements have to have the same size");
        }
      }
    }
  }
}

#if HAVE_DUNE_UDG
template <int dim>
static duneuro::UDGMEEGDriverData<dim> extractUDGDataFromMainDict(py::dict d)
{
  duneuro::UDGMEEGDriverData<dim> data;
  if (d.contains("domain") && d.contains("level_sets")) {
    auto domainDict = d["domain"].cast<py::dict>();
    auto list = domainDict["level_sets"].cast<py::list>();
    for (auto lvlst : list) {
      auto levelsetDict = domainDict[lvlst].cast<py::dict>();
      if (levelsetDict["type"].cast<py::str>() == py::str("image")) {
        for (auto item : levelsetDict) {
          if (item.first.cast<py::str>() == py::str("data")) {
            auto buffer = item.second.cast<py::buffer>();
            py::buffer_info info = buffer.request();
            if (info.format != py::format_descriptor<double>::value) {
              DUNE_THROW(Dune::Exception, "only float level sets are supported. expected "
                                              << py::format_descriptor<double>::value << " got "
                                              << info.format);
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
                          docstr.str().c_str())
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
      .def("__init__",
           [](FieldVector& instance, py::array_t<T> array) {
             if (array.size() != dim)
               DUNE_THROW(Dune::Exception, "array has to have size " << dim << " but got "
                                                                     << array.size());
             if (array.ndim() != 1)
               DUNE_THROW(Dune::Exception, "array has to have dim " << 1 << " but got "
                                                                    << array.ndim());
             new (&instance) FieldVector();
             std::copy(array.data(), array.data() + dim, instance.begin());
           })
      .def(py::self += py::self)
      .def(py::self -= py::self)
      .def(py::self += T())
      .def(py::self -= T())
      .def(py::self *= T())
      .def(py::self /= T())
      .def("__len__", &FieldVector::size)
      .def("__getitem__",
           [](const FieldVector& instance, std::size_t i) {
             return assert(i < dim);
             instance[i];
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
      .def("__init__",
           [](Dipole& instance, py::array_t<T> pos, py::array_t<T> mom) {
             if (pos.size() != dim || pos.ndim() != 1)
               DUNE_THROW(Dune::Exception, "position has to have " << dim << " entries");
             if (mom.size() != dim || pos.ndim() != 1)
               DUNE_THROW(Dune::Exception, "moment has to have " << dim << " entries");
             FieldVector vpos, vmom;
             std::copy(pos.data(), pos.data() + dim, vpos.begin());
             std::copy(mom.data(), mom.data() + dim, vmom.begin());
             new (&instance) Dipole(vpos, vmom);
           },
           "create a dipole from its position and moment", py::arg("position"), py::arg("moment"))
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

template <int dim>
class PyMEEGDriverInterface
{
public:
  using Interface = duneuro::MEEGDriverInterface<dim>;
  explicit PyMEEGDriverInterface(py::dict d)
  {
    duneuro::MEEGDriverData<dim> data;
#if HAVE_DUNE_UDG
    data.udgData = extractUDGDataFromMainDict<dim>(d);
    extractFittedDataFromMainDict(d, data.fittedData);
#endif
    driver_ = duneuro::MEEGDriverFactory<dim>::make_meeg_driver(dictToParameterTree(d), data);
  }

  std::unique_ptr<duneuro::Function> makeDomainFunction() const
  {
    return driver_->makeDomainFunction();
  }

  py::dict solveEEGForward(const typename Interface::DipoleType& dipole,
                           duneuro::Function& solution, py::dict config)
  {
    auto storage = std::make_shared<ParameterTreeStorage>();
    driver_->solveEEGForward(dipole, solution, dictToParameterTree(config),
                             duneuro::DataTree(storage));
    return toPyDict(storage->tree);
  }

  std::pair<std::vector<double>, py::dict> solveMEGForward(const duneuro::Function& eegSolution,
                                                           py::dict config)
  {
    auto storage = std::make_shared<ParameterTreeStorage>();
    auto result = driver_->solveMEGForward(eegSolution, dictToParameterTree(config),
                                           duneuro::DataTree(storage));
    return {result, toPyDict(storage->tree)};
  }

  py::dict write(const duneuro::Function& solution, py::dict config) const
  {
    auto storage = std::make_shared<ParameterTreeStorage>();
    driver_->write(solution, dictToParameterTree(config), duneuro::DataTree(storage));
    return toPyDict(storage->tree);
  }

  py::dict write(py::dict config) const
  {
    auto storage = std::make_shared<ParameterTreeStorage>();
    driver_->write(dictToParameterTree(config), duneuro::DataTree(storage));
    return toPyDict(storage->tree);
  }

  void setElectrodes(const std::vector<typename Interface::CoordinateType>& electrodes,
                     py::dict config)
  {
    driver_->setElectrodes(electrodes, dictToParameterTree(config));
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
    auto storage = std::make_shared<ParameterTreeStorage>();
    std::unique_ptr<duneuro::DenseMatrix<double>> result =
        driver_->computeEEGTransferMatrix(dictToParameterTree(config), duneuro::DataTree(storage));
    return {result.release(), toPyDict(storage->tree)};
  }

  std::pair<duneuro::DenseMatrix<double>*, py::dict> computeMEGTransferMatrix(py::dict config)
  {
    auto storage = std::make_shared<ParameterTreeStorage>();
    std::unique_ptr<duneuro::DenseMatrix<double>> result =
        driver_->computeMEGTransferMatrix(dictToParameterTree(config), duneuro::DataTree(storage));
    return {result.release(), toPyDict(storage->tree)};
  }

  std::pair<std::vector<double>, py::dict>
  applyEEGTransfer(py::buffer buffer, const typename Interface::DipoleType& dipole, py::dict config)
  {
    auto transferMatrix = toDenseMatrix(buffer);
    auto storage = std::make_shared<ParameterTreeStorage>();
    auto result = driver_->applyEEGTransfer(*transferMatrix, dipole, dictToParameterTree(config),
                                            duneuro::DataTree(storage));
    return {result, toPyDict(storage->tree)};
  }

  std::pair<std::vector<double>, py::dict>
  applyMEGTransfer(py::buffer buffer, const typename Interface::DipoleType& dipole, py::dict config)
  {
    auto transferMatrix = toDenseMatrix(buffer);
    auto storage = std::make_shared<ParameterTreeStorage>();
    auto result = driver_->applyMEGTransfer(*transferMatrix, dipole, dictToParameterTree(config),
                                            duneuro::DataTree(storage));
    return {result, toPyDict(storage->tree)};
  }

private:
  std::unique_ptr<Interface> driver_;
  Dune::ParameterTree tree_;
};

template <int dim>
static inline void register_meeg_driver_interface(py::module& m)
{
  using Interface = PyMEEGDriverInterface<dim>;
  using write1 = py::dict (Interface::*)(const duneuro::Function&, py::dict) const;
  using write2 = py::dict (Interface::*)(py::dict) const;
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
      .def("write", write1(&Interface::write))
      .def("write", write2(&Interface::write))
      .def("setElectrodes", &Interface::setElectrodes,
           "set the electrodes. subsequent calls to evaluateAtElectrodes will use these "
           "electrodes.",
           py::arg("electrodes"), py::arg("config"))
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
           py::arg("matrix"), py::arg("dipole"), py::arg("config"))
      .def("applyMEGTransfer", &Interface::applyMEGTransfer, "apply the meg transfer matrix",
           py::arg("matrix"), py::arg("dipole"), py::arg("config"));
  ;
}

template <class T>
void register_dense_matrix(py::module& m)
{
  using Matrix = duneuro::DenseMatrix<T>;
  py::class_<Matrix>(m, "Matrix").def_buffer([](Matrix& m) -> py::buffer_info {
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
          auto ptree = dictToParameterTree(d);
          return duneuro::generate_points_on_sphere<double, dim>(ptree);
        },
        R"pydoc(
generate approximately uniformly distributed points on a sphere

:param config: configuration for point generation
)pydoc",
        py::arg("config"));
}

static inline void register_analytical_solution(py::module& m)
{
  m.def("analytical_solution_3d",
        [](const std::vector<Dune::FieldVector<double, 3>>& electrodes,
           const duneuro::Dipole<double, 3>& dipole, py::dict config) {
          return duneuro::compute_analytic_solution(electrodes, dipole,
                                                    dictToParameterTree(config));
        },
        R"pydoc(
compute the analytical solution of the EEG forward problem
  )pydoc");
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
  py::class_<PS>(m, name.str().c_str()).def("__init__", [](PS& instance, py::dict d) {
    new (&instance) PS(dictToParameterTree(d));
  });
}

template <int dim>
class PyTDCSDriverInterface
{
public:
  using Interface = duneuro::TDCSDriverInterface<dim>;
  explicit PyTDCSDriverInterface(py::dict d)
  {
    duneuro::TDCSDriverData<dim> data;
    extractFittedDataFromMainDict(d, data.fittedData);
    driver_ = duneuro::TDCSDriverFactory<dim>::make_tdcs_driver(dictToParameterTree(d), data);
  }

  std::unique_ptr<duneuro::Function> makeDomainFunction() const
  {
    return driver_->makeDomainFunction();
  }

  py::dict write(const duneuro::Function& solution, py::dict config) const
  {
    auto storage = std::make_shared<ParameterTreeStorage>();
    driver_->write(solution, dictToParameterTree(config), duneuro::DataTree(storage));
    return toPyDict(storage->tree);
  }

  py::dict solveTDCSForward(const duneuro::PatchSet<double, dim>& patchSet,
                            duneuro::Function& solution, py::dict config) const
  {
    auto storage = std::make_shared<ParameterTreeStorage>();
    driver_->solveTDCSForward(patchSet, solution, dictToParameterTree(config),
                              duneuro::DataTree(storage));
    return toPyDict(storage->tree);
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
      .def(py::init<py::dict>())
      .def("makeDomainFunction", &Interface::makeDomainFunction, "create a domain function")
      .def("write", &Interface::write)
      .def("solveTDCSForward", &Interface::solveTDCSForward);
}

PYBIND11_PLUGIN(duneuropy)
{
  py::module m("duneuropy", "duneuropy library");

  register_exceptions();

  register_function(m);

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

  register_analytical_solution(m);

  return m.ptr();
}
