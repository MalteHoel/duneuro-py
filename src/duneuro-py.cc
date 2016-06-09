#if HAVE_CONFIG_H
#include <config.h>
#endif

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <dune/common/parametertree.hh>

#include <duneuro/common/dense_matrix.hh>
#include <duneuro/eeg/eeg_driver_factory.hh>
#include <duneuro/eeg/eeg_driver_interface.hh>
#include <duneuro/io/dipole_reader.hh>
#include <duneuro/io/field_vector_reader.hh>

namespace py = pybind11;
// translate a dune exception to std::exception since the latter is automatically converted to a
// python error
struct DubioException : public std::exception {
  explicit DubioException(const Dune::Exception& ex) : what_(ex.what())
  {
  }
  const char* what() const noexcept override
  {
    return what_.c_str();
  }
  std::string what_;
};

// translate a python dict to a Dune::ParameterTree. The dict can be nested and the sub dicts will
// be translated recursively. Note that the type of the values is assumed to be str
Dune::ParameterTree dictToParameterTree(const py::dict& dict)
{
  Dune::ParameterTree tree;
  for (const auto& item : dict) {
    std::string type = py::object(item.second.get_type().attr("__name__")).cast<std::string>();
    if (type == "str") {
      tree[item.first.cast<std::string>()] = item.second.cast<std::string>();
    } else if (type == "dict") {
      auto subtree = dictToParameterTree(py::dict(item.second, true));
      tree.sub(item.first.cast<std::string>()) = subtree;
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
      .def(py::init<T>())
      .def("__init__",
           [](FieldVector& instance, const py::tuple& data) {
             FieldVector vec;
             for (unsigned int i = 0; i < dim; ++i) {
               vec[i] = py::object(data[i]).cast<T>();
             }
             new (&instance) FieldVector(vec);
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
      .def("__init__",
           [](Dipole& instance, const py::tuple& position, const py::tuple& moment) {
             FieldVector pos, mom;
             for (unsigned int i = 0; i < dim; ++i) {
               pos[i] = py::object(position[i]).cast<T>();
               mom[i] = py::object(moment[i]).cast<T>();
             }
             new (&instance) Dipole(pos, mom);
           },
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

void register_function(py::module& m)
{
  py::class_<duneuro::Function>(m, "Function").def("__init__", [](duneuro::Function& instance) {
    new (&instance) duneuro::Function(std::shared_ptr<double>(nullptr));
  });
}

class PyEEGDriverInterface
{
public:
  duneuro::Function makeDomainFunction() const
  {
    PYBIND11_OVERLOAD_PURE(duneuro::Function, duneuro::EEGDriverInterface, makeDomainFunction);
  }

  void solve(const duneuro::EEGDriverInterface::DipoleType& dipole, duneuro::Function& solution)
  {
    PYBIND11_OVERLOAD_PURE(void, duneuro::EEGDriverInterface, solve, dipole, solution);
  }

  void write(const Dune::ParameterTree& config, const duneuro::Function& solution) const
  {
    PYBIND11_OVERLOAD_PURE(void, duneuro::EEGDriverInterface, write, config, solution);
  }

  void setElectrodes(const std::vector<duneuro::EEGDriverInterface::CoordinateType>& electrodes)
  {
    PYBIND11_OVERLOAD_PURE(void, duneuro::EEGDriverInterface, setElectrodes, electrodes);
  }

  std::vector<double> evaluateAtElectrodes(const duneuro::Function& solution) const
  {
    PYBIND11_OVERLOAD_PURE(std::vector<double>, duneuro::EEGDriverInterface, evaluateAtElectrodes,
                           solution);
  }

  duneuro::DenseMatrix<double>* computeTransferMatrix()
  {
    PYBIND11_OVERLOAD_PURE(duneuro::DenseMatrix<double>*, duneuro::EEGDriverInterface,
                           computeTransferMatrix);
  }

  std::vector<double> solve(const duneuro::DenseMatrix<double>& transferMatrix,
                            const duneuro::EEGDriverInterface::DipoleType& dipole)
  {
    PYBIND11_OVERLOAD_PURE(std::vector<double>, duneuro::EEGDriverInterface, solve, transferMatrix,
                           dipole);
  }
};

void register_eeg_driver_interface(py::module& m)
{
  py::class_<PyEEGDriverInterface> driver(m, "EEGDriver");
  driver.alias<duneuro::EEGDriverInterface>()
      .def(py::init<>())
      .def("makeDomainFunction", &duneuro::EEGDriverInterface::makeDomainFunction)
      .def("solve",
           [](duneuro::EEGDriverInterface& interface, const duneuro::Dipole<double, 3>& dipole,
              duneuro::Function& solution) { interface.solve(dipole, solution); })
      .def("solve",
           [](duneuro::EEGDriverInterface& interface, const duneuro::Dipole<double, 3>& dipole) {
             auto solution = interface.makeDomainFunction();
             interface.solve(dipole, solution);
             return solution;
           })
      .def("write",
           [](duneuro::EEGDriverInterface& interface, const py::dict& config,
              const duneuro::Function& solution) {
             auto ptree = dictToParameterTree(config);
             interface.write(ptree, solution);
           })
      .def("setElectrodes", &duneuro::EEGDriverInterface::setElectrodes)
      .def("evaluateAtElectrodes", &duneuro::EEGDriverInterface::evaluateAtElectrodes)
      .def("computeTransferMatrix",
           [](duneuro::EEGDriverInterface& interface) {
             return interface.computeTransferMatrix().release();
           })
      .def("solve", [](duneuro::EEGDriverInterface& interface, py::buffer& buffer,
                       const duneuro::EEGDriverInterface::DipoleType& dipole) {
        /* Request a buffer descriptor from Python */
        py::buffer_info info = buffer.request();

        /* Some sanity checks ... */
        if (info.format != py::format_descriptor<double>::value())
          throw std::runtime_error("Incompatible format: expected a double array!");

        if (info.ndim != 2)
          throw std::runtime_error("Incompatible buffer dimension!");

        if (info.strides[1] / sizeof(double) != 1)
          throw std::runtime_error("Supporting only row major format");

        duneuro::DenseMatrix<double> transferMatrix(info.shape[0], info.shape[1]);
        double* ptr = static_cast<double*>(info.ptr);
        std::copy(ptr, ptr + info.shape[0] * info.shape[1], transferMatrix.data());

        return interface.solve(transferMatrix, dipole);
      });
  ;
}

void register_eeg_driver_factory(py::module& m)
{
  m.def("make_eeg_driver", [](const py::dict& dict) {
    try {
      auto ptree = dictToParameterTree(dict);
      return duneuro::EEGDriverFactory::make_eeg_driver(ptree).release();
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

PYBIND11_PLUGIN(duneuropy)
{
  py::module m("duneuropy", "duneuropy library");

  register_field_vector<double, 3>(m);
  register_dipole<double, 3>(m);
  register_read_dipoles<double, 3>(m);

  register_field_vector_reader<double, 3>(m);

  register_function(m);

  register_eeg_driver_interface(m);

  register_eeg_driver_factory(m);

  register_dense_matrix<double>(m);

  return m.ptr();
}
