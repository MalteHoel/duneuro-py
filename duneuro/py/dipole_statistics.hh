#ifndef DUNEURO_PY_DIPOLE_STATISTICS_HH
#define DUNEURO_PY_DIPOLE_STATISTICS_HH

#include <pybind11/pybind11.h>

#include <duneuro/common/dipole_statistics.hh>
#include <duneuro/common/dipole_statistics_factory.hh>
#include <duneuro/py/fitted_data.hh>
#include <duneuro/py/parameter_tree.h>

namespace duneuro
{
  template <int dim>
  class PyDipoleStatistics
  {
  public:
    explicit PyDipoleStatistics(pybind11::dict d)
    {
      FittedDriverData<dim> data;
      extractFittedDataFromMainDict(d, data);
      statistics_ = DipoleStatisticsFactory<dim>::make_dipole_statistics(toParameterTree(d), data);
    }

    std::array<std::array<double, dim>, dim> conductivity(const Dipole<double, dim>& x)
    {
      std::array<std::array<double, dim>, dim> result;
      auto c = statistics_->conductivity(x);
      for (unsigned int i = 0; i < dim; ++i) {
        for (unsigned int j = 0; j < dim; ++j) {
          result[i][j] = c[i][j];
        }
      }
      return result;
    }

  private:
    std::unique_ptr<DipoleStatisticsInterface<dim>> statistics_;
  };

  template <int dim>
  void register_dipole_statistics(pybind11::module& m)
  {
    using C = PyDipoleStatistics<dim>;
    std::string classname = "DipoleStatistics" + std::to_string(dim) + "D";
    pybind11::class_<C>(m, classname.c_str())
        .def(pybind11::init<pybind11::dict>())
        .def("conductivity", &C::conductivity);
  }
}

#endif // DUNEURO_PY_DIPOLE_STATISTICS_HH
