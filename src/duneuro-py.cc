#if HAVE_CONFIG_H
#include <config.h>
#endif

#include <Python.h>

#include <memory>

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
#include <duneuro/io/projections_reader.hh>
#include <duneuro/meeg/meeg_driver_factory.hh>
#include <duneuro/meeg/meeg_driver_interface.hh>

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
    out[py::str(key)] = py::str(tree[key]);
  }
  for (const auto& subkey : tree.getSubKeys()) {
    out[py::str(subkey)] = toPyDict(tree.sub(subkey));
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
  else if (type == "list") {
    std::vector<std::string> entries;
    for (auto it = handle.begin(); it != handle.end(); ++it)
      entries.push_back(py_to_string(*it));
    if (entries.size() > 0) {
      std::stringstream str;
      str << entries[0];
      for (unsigned int i = 1; i < entries.size(); ++i) {
        str << " " << entries[i];
      }
      return str.str();
    } else {
      return "";
    }
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

static duneuro::UDGMEEGDriverData extractUDGDataFromMainDict(py::dict d)
{
  duneuro::UDGMEEGDriverData data;
  try {
    auto domainDict = d[py::str("domain")].cast<py::dict>();
    auto list = domainDict[py::str("level_sets")].cast<py::list>();
    for (auto lvlst : list) {
      auto levelsetDict = domainDict[lvlst].cast<py::dict>();
      if (levelsetDict[py::str("type")] == py::str("image")) {
        for (auto item : levelsetDict) {
          if (item.first.cast<std::string>() == "data") {
            std::string type = item.second.get_type().attr("__name__").cast<std::string>();
            auto buffer = item.second.cast<py::buffer>();
            py::buffer_info info = buffer.request();
            if (info.ndim != 3) {
              DUNE_THROW(Dune::Exception, "only 3d level sets are supported");
            }
            if (info.format != py::format_descriptor<double>::value) {
              DUNE_THROW(Dune::Exception, "only float level sets are supported. expected "
                                              << py::format_descriptor<double>::value << " got "
                                              << info.format);
            }
            duneuro::SimpleStructuredGrid<3> grid({static_cast<unsigned int>(info.shape[0]),
                                                   static_cast<unsigned int>(info.shape[1]),
                                                   static_cast<unsigned int>(info.shape[2])});
            double* ptr = reinterpret_cast<double*>(info.ptr);
            auto imagedata = std::make_shared<std::vector<double>>(ptr, ptr + info.size);
            data.levelSetData.images.push_back(
                std::make_shared<duneuro::Image<double, 3>>(imagedata, grid));
            levelsetDict[py::str("image_index")] = py::int_(data.levelSetData.images.size() - 1);
          }
        }
      }
    }
  } catch (py::index_error& ex) {
  } catch (py::cast_error& ex) {
  }
  return data;
}

// create basic binding for a dune field vector
template <class T, int dim>
void register_field_vector(py::module& m)
{
  using FieldVector = Dune::FieldVector<T, dim>;
  py::class_<FieldVector>(m, (std::string("FieldVector") + std::to_string(dim) + "D").c_str())
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
      .def("__init__", [](FieldVector& instance, py::buffer buffer) {
        /* Request a buffer descriptor from Python */
        py::buffer_info info = buffer.request();

        /* Some sanity checks ... */
        if (info.format != py::format_descriptor<T>::value)
          throw std::runtime_error("Incompatible format: expected a T array!");

        if (info.ndim != 1)
          throw std::runtime_error("Incompatible buffer dimension!");

        if (info.shape[0] != dim)
          throw std::runtime_error("Incompartible buffer size");

        T* ptr = static_cast<T*>(info.ptr);

        std::copy(ptr, ptr + dim, instance.begin());
      });
  //.def(py::self += py::self)
  //.def(py::self -= py::self)
  //.def(py::self += T())
  //.def(py::self -= T())
  //.def(py::self *= T())
  //.def(py::self /= T())
  //.def("__len__", &FieldVector::size)
  //.def("__getitem__", [](FieldVector& instance, std::size_t i) { return instance[i]; })
  //.def("__str__", [](const FieldVector& fv) {
  // std::stringstream sstr;
  // sstr << fv;
  // return sstr.str();
  //});
}

// create bindings for a dipole
template <class T, int dim>
void register_dipole(py::module& m)
{
  using Dipole = duneuro::Dipole<T, dim>;
  using FieldVector = Dune::FieldVector<T, dim>;
  py::class_<Dipole>(m, "Dipole")
      .def(py::init<FieldVector, FieldVector>(), "create a dipole from its position and moment",
           py::arg("position"), py::arg("moment"))
      .def("position", &Dipole::position, "position", py::return_value_policy::reference_internal)
      .def("moment", &Dipole::moment, "moment", py::return_value_policy::reference_internal);
  //.def("__str__", [](const Dipole& dip) {
  // std::stringstream sstr;
  // sstr << "position: " << dip.position() << " moment: " << dip.moment();
  // return sstr.str();
  //});
}

template <class T, int dim>
void register_read_dipoles(py::module& m)
{
  m.def("read_dipoles",
        [](const std::string& filename) { return duneuro::DipoleReader<T, dim>::read(filename); });
}

template <class T, int dim>
void register_field_vector_reader(py::module& m)
{
  auto name = "read_" + std::to_string(dim) + "d_field_vectors";
  m.def(name.c_str(), [](const std::string& filename) {
    return duneuro::FieldVectorReader<T, dim>::read(filename);
  });
}

template <class T, int dim>
void register_projections_reader(py::module& m)
{
  auto name = "read_" + std::to_string(dim) + "d_projections";
  m.def(name.c_str(), [](const std::string& filename) {
    return duneuro::ProjectionsReader<T, dim>::read(filename);
  });
}

static inline void register_function(py::module& m)
{
  py::class_<duneuro::Function>(m, "FunctionWrapper");
}

class PyMEEGDriverInterface
{
public:
  explicit PyMEEGDriverInterface(py::dict d)
  {
    auto data = extractUDGDataFromMainDict(d);
    dictToParameterTree(d).report(std::cout);
    driver_ = duneuro::MEEGDriverFactory::make_meeg_driver(dictToParameterTree(d),
                                                           duneuro::MEEGDriverData{data});
  }

  std::unique_ptr<duneuro::Function> makeDomainFunction() const
  {
    return driver_->makeDomainFunction();
  }

  py::dict solveEEGForward(const duneuro::MEEGDriverInterface::DipoleType& dipole,
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

  void setElectrodes(const std::vector<duneuro::MEEGDriverInterface::CoordinateType>& electrodes,
                     py::dict config)
  {
    driver_->setElectrodes(electrodes, dictToParameterTree(config));
  }

  std::vector<double> evaluateAtElectrodes(const duneuro::Function& solution) const
  {
    return driver_->evaluateAtElectrodes(solution);
  }

  void setCoilsAndProjections(
      const std::vector<duneuro::MEEGDriverInterface::CoordinateType>& coils,
      const std::vector<std::vector<duneuro::MEEGDriverInterface::CoordinateType>>& projections)
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
  applyEEGTransfer(py::buffer buffer, const duneuro::MEEGDriverInterface::DipoleType& dipole,
                   py::dict config)
  {
    auto transferMatrix = toDenseMatrix(buffer);
    auto storage = std::make_shared<ParameterTreeStorage>();
    auto result = driver_->applyEEGTransfer(*transferMatrix, dipole, dictToParameterTree(config),
                                            duneuro::DataTree(storage));
    return {result, toPyDict(storage->tree)};
  }

  std::pair<std::vector<double>, py::dict>
  applyMEGTransfer(py::buffer buffer, const duneuro::MEEGDriverInterface::DipoleType& dipole,
                   py::dict config)
  {
    auto transferMatrix = toDenseMatrix(buffer);
    auto storage = std::make_shared<ParameterTreeStorage>();
    auto result = driver_->applyMEGTransfer(*transferMatrix, dipole, dictToParameterTree(config),
                                            duneuro::DataTree(storage));
    return {result, toPyDict(storage->tree)};
  }

private:
  std::unique_ptr<duneuro::MEEGDriverInterface> driver_;
  Dune::ParameterTree tree_;
};

static inline void register_meeg_driver_interface(py::module& m)
{
  using write1 = py::dict (PyMEEGDriverInterface::*)(const duneuro::Function&, py::dict) const;
  using write2 = py::dict (PyMEEGDriverInterface::*)(py::dict) const;
  py::class_<PyMEEGDriverInterface>(m, "MEEGDriver")
      .def(py::init<py::dict>())
      .def("makeDomainFunction", &PyMEEGDriverInterface::makeDomainFunction /* , */
           /* "create a domain function" */)
      .def("solveEEGForward", &PyMEEGDriverInterface::solveEEGForward
           /* , */
           /* "solve the eeg forward problem and store the result in the given function" */)
      .def("solveMEGForward", &PyMEEGDriverInterface::solveMEGForward
           /* , */
           /* "solve the meg forward problem and return the solution" */)
      .def("write", write1(&PyMEEGDriverInterface::write))
      .def("write", write2(&PyMEEGDriverInterface::write))
      .def("setElectrodes", &PyMEEGDriverInterface::setElectrodes /* , */
           // "set the electrodes. subsequent calls to evaluateAtElectrodes will use these "
           /* "electrodes." */)
      .def("setCoilsAndProjections", &PyMEEGDriverInterface::setCoilsAndProjections
           /* , */
           // "set coils and projections for meg. subsequent calls to solveMEGForward will use "
           // "these "
           /* "coils and projections" */)
      .def("evaluateAtElectrodes", &PyMEEGDriverInterface::evaluateAtElectrodes /* , */
           /* "evaluate the given function at the set electrodes" */)
      .def("computeEEGTransferMatrix", &PyMEEGDriverInterface::computeEEGTransferMatrix
           /* , */
           /* "compute the eeg transfer matrix using the set electrodes" */)
      .def("computeMEGTransferMatrix", &PyMEEGDriverInterface::computeMEGTransferMatrix
           /* , */
           /* "compute the meg transfer matrix using the set coils and projections" */)
      .def("applyEEGTransfer", &PyMEEGDriverInterface::applyEEGTransfer)
      .def("applyMEGTransfer", &PyMEEGDriverInterface::applyMEGTransfer);
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

static inline void register_points_on_sphere(py::module& m)
{
  m.def("generate_points_on_sphere",
        [](py::dict d) {
          auto ptree = dictToParameterTree(d);
          return duneuro::generate_points_on_sphere<double, 3>(ptree);
        } /* , */
        /* "generate approximately uniformly distributed points on a sphere" */);
}

static inline void register_analytical_solution(py::module& m)
{
  m.def("analytical_solution", [](const std::vector<Dune::FieldVector<double, 3>>& electrodes,
                                  const duneuro::Dipole<double, 3>& dipole, py::dict config) {
    return duneuro::compute_analytic_solution(electrodes, dipole, dictToParameterTree(config));
  });
}

PYBIND11_PLUGIN(duneuropy)
{
  py::module m("duneuropy", "duneuropy library");

  register_exceptions();

  register_field_vector<double, 3>(m);
  register_dipole<double, 3>(m);
  register_read_dipoles<double, 3>(m);

  register_field_vector_reader<double, 3>(m);
  register_projections_reader<double, 3>(m);

  register_function(m);

  register_dense_matrix<double>(m);

  register_meeg_driver_interface(m);

  register_points_on_sphere(m);

  register_analytical_solution(m);

  return m.ptr();
}
