#if HAVE_CONFIG_H
#include <config.h>
#endif

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <dune/common/parametertree.hh>
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
// translate a dune exception to std::exception since the latter is automatically converted to a
// python error
struct DubioException : public std::exception {
  explicit DubioException(const std::string& msg) : what_(msg)
  {
  }

  explicit DubioException(const Dune::Exception& ex) : what_(ex.what())
  {
  }
  const char* what() const noexcept override
  {
    return what_.c_str();
  }
  std::string what_;
};

struct PyDictStorage : public duneuro::StorageInterface {
  virtual void store(const std::string& name, const std::string& value)
  {
    dict[py::str(name)] = py::str(value);
  }

  virtual void storeMatrix(const std::string& name,
                           std::shared_ptr<duneuro::MatrixInterface<double>> matrix)
  {
  }

  virtual void storeMatrix(const std::string& name,
                           std::shared_ptr<duneuro::MatrixInterface<unsigned int>> matrix)
  {
  }

  py::dict dict;
};

std::string py_to_string(const py::handle& handle)
{
  std::string type = py::object(handle.get_type().attr("__name__")).cast<std::string>();
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
    std::transform(handle.begin(), handle.end(), std::back_inserter(entries), py_to_string);
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
  throw DubioException(std::string("type \"") + type + std::string("\" not supported"));
}

// translate a python dict to a Dune::ParameterTree. The dict can be nested and the sub dicts will
// be translated recursively. Note that the type of the values is assumed to be str
Dune::ParameterTree dictToParameterTree(const py::dict& dict)
{
  Dune::ParameterTree tree;
  for (const auto& item : dict) {
    std::string type = py::object(item.second.get_type().attr("__name__")).cast<std::string>();
    if (type == "dict") {
      auto subtree = dictToParameterTree(py::dict(item.second, true));
      tree.sub(py_to_string(item.first)) = subtree;
    } else {
      tree[py_to_string(item.first)] = py_to_string(item.second);
    }
  }
  return tree;
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
            py::format_descriptor<T>::value(), /* Python struct-style format descriptor */
            1, /* Number of dimensions */
            {dim}, /* Buffer dimensions */
            {sizeof(T)});
      })
      .def(py::init<T>())
      .def("__init__",
           [](FieldVector& instance, py::buffer buffer) {
             /* Request a buffer descriptor from Python */
             py::buffer_info info = buffer.request();

             /* Some sanity checks ... */
             if (info.format != py::format_descriptor<T>::value())
               throw std::runtime_error("Incompatible format: expected a T array!");

             if (info.ndim != 1)
               throw std::runtime_error("Incompatible buffer dimension!");

             if (info.shape[0] != dim)
               throw std::runtime_error("Incompartible buffer size");

             T* ptr = static_cast<T*>(info.ptr);

             std::copy(ptr, ptr + dim, instance.begin());
           })
      .def(py::self += py::self)
      .def(py::self -= py::self)
      .def(py::self += T())
      .def(py::self -= T())
      .def(py::self *= T())
      .def(py::self /= T())
      .def("__len__", &FieldVector::size)
      .def("__getitem__", [](FieldVector& instance, std::size_t i) { return instance[i]; })
      .def("__str__", [](const FieldVector& fv) {
        std::stringstream sstr;
        sstr << fv;
        return sstr.str();
      });
}

// create bindings for a dipole
template <class T, int dim>
void register_dipole(py::module& m)
{
  using Dipole = duneuro::Dipole<T, dim>;
  using FieldVector = Dune::FieldVector<T, dim>;
  py::class_<Dipole>(m, "Dipole")
      .def(py::init<const FieldVector&, const FieldVector&>(),
           "create a dipole from its position and moment", py::arg("position"), py::arg("moment"))
      .def("position", [](const Dipole& dip) { return dip.position(); }, "position",
           py::return_value_policy::copy)
      .def("moment", [](const Dipole& dip) { return dip.moment(); }, "moment",
           py::return_value_policy::copy)
      .def("__str__", [](const Dipole& dip) {
        std::stringstream sstr;
        sstr << "position: " << dip.position() << " moment: " << dip.moment();
        return sstr.str();
      });
}

template <class F>
py::dict output_call(F func)
{
  auto storage = std::make_shared<PyDictStorage>();
  auto res = func(duneuro::DataTree(storage));

  py::dict result;
  result[py::str("config")] = storage->dict;
  result[py::str("result")] = py::cast(res.release());
  return result;
}

template <class F>
py::dict void_call(F func)
{
  auto storage = std::make_shared<PyDictStorage>();
  func(duneuro::DataTree(storage));
  py::dict result;
  result[py::str("config")] = storage->dict;
  return result;
}

template <class T, int dim>
void register_read_dipoles(py::module& m)
{
  m.def("read_dipoles", [](const std::string& filename) {
    return output_call([&](duneuro::DataTree dtree) {
      auto res = Dune::Std::make_unique<std::vector<duneuro::Dipole<T, dim>>>();
      duneuro::DipoleReader<T, dim>::read(filename, std::back_inserter(*res), dtree);
      return std::move(res);
    });
  });
}

template <class T, int dim>
void register_field_vector_reader(py::module& m)
{
  auto name = "read_" + std::to_string(dim) + "d_field_vectors";
  m.def(name.c_str(), [](const std::string& filename) {
    return output_call([&](duneuro::DataTree dtree) {
      auto res = Dune::Std::make_unique<std::vector<Dune::FieldVector<T, dim>>>();
      duneuro::FieldVectorReader<T, dim>::read(filename, std::back_inserter(*res), dtree);
      return std::move(res);
    });
  });
}

template <class T, int dim>
void register_projections_reader(py::module& m)
{
  auto name = "read_" + std::to_string(dim) + "d_projections";
  m.def(name.c_str(), [](const std::string& filename) {
    return output_call([&](duneuro::DataTree dtree) {
      auto res = Dune::Std::make_unique<std::vector<std::vector<Dune::FieldVector<T, dim>>>>();
      duneuro::ProjectionsReader<T, dim>::read(filename, std::back_inserter(*res), dtree);
      return std::move(res);
    });
  });
}

void register_function(py::module& m)
{
  py::class_<duneuro::Function>(m, "Function").def("__init__", [](duneuro::Function& instance) {
    new (&instance) duneuro::Function(std::shared_ptr<double>(nullptr));
  });
}

class PyMEEGDriverInterface
{
public:
  duneuro::Function makeDomainFunction() const
  {
    PYBIND11_OVERLOAD_PURE(duneuro::Function, duneuro::EEGDriverInterface, makeDomainFunction);
  }

  void solveEEGForward(const duneuro::MEEGDriverInterface::DipoleType& dipole,
                       duneuro::Function& solution)
  {
    PYBIND11_OVERLOAD_PURE(void, duneuro::MEEGDriverInterface, solveEEGForward, dipole, solution);
  }

  std::vector<double> solveMEGForward(const duneuro::Function& eegSolution)
  {
    PYBIND11_OVERLOAD_PURE(std::vector<double>, duneuro::MEEGDriverInterface, solveMEGForward,
                           eegSolution);
  }

  void write(const Dune::ParameterTree& config, const duneuro::Function& solution) const
  {
    PYBIND11_OVERLOAD_PURE(void, duneuro::MEEGDriverInterface, write, config, solution);
  }

  void setElectrodes(const std::vector<duneuro::MEEGDriverInterface::CoordinateType>& electrodes)
  {
    PYBIND11_OVERLOAD_PURE(void, duneuro::EEGDriverInterface, setElectrodes, electrodes);
  }

  std::vector<double> evaluateAtElectrodes(const duneuro::Function& solution) const
  {
    PYBIND11_OVERLOAD_PURE(std::vector<double>, duneuro::MEEGDriverInterface, evaluateAtElectrodes,
                           solution);
  }

  void setCoilsAndProjections(
      const std::vector<duneuro::MEEGDriverInterface::CoordinateType>& coils,
      const std::vector<std::vector<duneuro::MEEGDriverInterface::CoordinateType>>& projections)
  {
    PYBIND11_OVERLOAD_PURE(void, duneuro::MEEGDriverInterface, setCoilsAndProjections, coils,
                           projections);
  }

  duneuro::DenseMatrix<double>* computeEEGTransferMatrix()
  {
    PYBIND11_OVERLOAD_PURE(duneuro::DenseMatrix<double>*, duneuro::MEEGDriverInterface,
                           computeEEGTransferMatrix);
  }

  duneuro::DenseMatrix<double>* computeMEGTransferMatrix()
  {
    PYBIND11_OVERLOAD_PURE(duneuro::DenseMatrix<double>*, duneuro::MEEGDriverInterface,
                           computeMEGTransferMatrix);
  }

  std::vector<double> applyTransfer(const duneuro::DenseMatrix<double>& transferMatrix,
                                    const duneuro::MEEGDriverInterface::DipoleType& dipole)
  {
    PYBIND11_OVERLOAD_PURE(std::vector<double>, duneuro::MEEGDriverInterface, applyTransfer,
                           transferMatrix, dipole);
  }
};

void register_meeg_driver_interface(py::module& m)
{
  py::class_<PyMEEGDriverInterface> driver(m, "MEEGDriver");
  driver.alias<duneuro::MEEGDriverInterface>()
      .def(py::init<>())
      .def("makeDomainFunction", &duneuro::MEEGDriverInterface::makeDomainFunction)
      .def("solveEEGForward",
           [](duneuro::MEEGDriverInterface& interface, const duneuro::Dipole<double, 3>& dipole,
              duneuro::Function& solution) {
             return void_call([&](duneuro::DataTree dtree) {
               interface.solveEEGForward(dipole, solution, dtree);
             });
           })
      .def("solveEEGForward",
           [](duneuro::MEEGDriverInterface& interface, const duneuro::Dipole<double, 3>& dipole) {
             return output_call([&](duneuro::DataTree dtree) {
               auto solution =
                   Dune::Std::make_unique<duneuro::Function>(interface.makeDomainFunction());
               interface.solveEEGForward(dipole, *solution, dtree);
               return std::move(solution);
             });
           })
      .def("solveMEGForward",
           [](duneuro::MEEGDriverInterface& interface, const duneuro::Function& eegSolution) {
             try {
               return output_call([&](duneuro::DataTree dtree) {
                 return Dune::Std::make_unique<std::vector<double>>(
                     interface.solveMEGForward(eegSolution, dtree));
               });
             } catch (Dune::Exception& ex) {
               throw DubioException(ex);
             }
           })
      .def("write",
           [](duneuro::MEEGDriverInterface& interface, const py::dict& config,
              const duneuro::Function& solution) {
             auto ptree = dictToParameterTree(config);
             interface.write(ptree, solution);
           })
      .def("setElectrodes", &duneuro::MEEGDriverInterface::setElectrodes)
      .def("setCoilsAndProjections",
           [](duneuro::MEEGDriverInterface& interface,
              const std::vector<duneuro::MEEGDriverInterface::CoordinateType>& coils,
              const std::vector<std::vector<duneuro::MEEGDriverInterface::CoordinateType>>&
                  projections) {
             try {
               interface.setCoilsAndProjections(coils, projections);
             } catch (Dune::Exception& ex) {
               throw DubioException(ex);
             }
           })
      .def("evaluateAtElectrodes", &duneuro::MEEGDriverInterface::evaluateAtElectrodes)
      .def("computeEEGTransferMatrix",
           [](duneuro::MEEGDriverInterface& interface) {
             return output_call([&](duneuro::DataTree dtree) {
               return interface.computeEEGTransferMatrix(dtree);
             });
           })
      .def("computeMEGTransferMatrix",
           [](duneuro::MEEGDriverInterface& interface) {
             return output_call([&](duneuro::DataTree dtree) {
               return interface.computeMEGTransferMatrix(dtree);
             });
           })
      .def("applyTransfer", [](duneuro::MEEGDriverInterface& interface, py::buffer& buffer,
                               const duneuro::MEEGDriverInterface::DipoleType& dipole) {
        return output_call([&](duneuro::DataTree dtree) {
          /* Request a buffer descriptor from Python */
          py::buffer_info info = buffer.request();

          /* Some sanity checks ... */
          if (info.format != py::format_descriptor<double>::value())
            throw std::runtime_error("Incompatible format: expected a double array!");

          if (info.ndim != 2)
            throw std::runtime_error("Incompatible buffer dimension!");

          if (info.strides[1] / sizeof(double) != 1)
            throw std::runtime_error("Supporting only row major format");

          duneuro::DenseMatrix<double> transferMatrix(info.shape[0], info.shape[1],
                                                      static_cast<double*>(info.ptr));
          return Dune::Std::make_unique<std::vector<double>>(
              interface.applyTransfer(transferMatrix, dipole, dtree));
        });
      });
  ;
}

void register_meeg_driver_factory(py::module& m)
{
  m.def("make_meeg_driver", [](const py::dict& dict) {
    try {
      auto ptree = dictToParameterTree(dict);
      return duneuro::MEEGDriverFactory::make_meeg_driver(ptree).release();
    } catch (Dune::Exception& ex) {
      throw DubioException(ex);
    }
  });
}

template <class T>
void register_dense_matrix(py::module& m)
{
  using Matrix = duneuro::DenseMatrix<T>;
  py::class_<Matrix>(m, "Matrix").def_buffer([](Matrix& m) -> py::buffer_info {
    return py::buffer_info(
        m.data(), /* Pointer to buffer */
        sizeof(T), /* Size of one scalar */
        py::format_descriptor<T>::value(), /* Python struct-style format descriptor */
        2, /* Number of dimensions */
        {m.rows(), m.cols()}, /* Buffer dimensions */
        {sizeof(T) * m.cols(), /* Strides (in bytes) for each index */
         sizeof(T)});
  });
}

void register_points_on_sphere(py::module& m)
{
  m.def("generate_points_on_sphere", [](const py::dict& d) {
    auto ptree = dictToParameterTree(d);
    try {
      return duneuro::generate_points_on_sphere<double, 3>(ptree);
    } catch (Dune::Exception& ex) {
      throw DubioException(ex);
    }
  });
}

void register_analytical_solution(py::module& m)
{
  m.def("analytical_solution",
        [](const std::vector<Dune::FieldVector<double, 3>>& electrodes,
           const duneuro::Dipole<double, 3>& dipole, const py::dict& config) {
          auto ptree = dictToParameterTree(config);
          return output_call([&](duneuro::DataTree dtree) {
            try {
              return Dune::Std::make_unique<std::vector<double>>(
                  duneuro::compute_analytic_solution(electrodes, dipole, ptree, dtree));
            } catch (Dune::Exception& ex) {
              throw DubioException(ex);
            }
          });
        });
}

PYBIND11_PLUGIN(duneuropy)
{
  py::module m("duneuropy", "duneuropy library");

  register_field_vector<double, 3>(m);
  register_dipole<double, 3>(m);
  register_read_dipoles<double, 3>(m);

  register_field_vector_reader<double, 3>(m);
  register_projections_reader<double, 3>(m);

  register_function(m);

  register_meeg_driver_interface(m);

  register_meeg_driver_factory(m);

  register_dense_matrix<double>(m);

  register_points_on_sphere(m);

  register_analytical_solution(m);

  return m.ptr();
}
