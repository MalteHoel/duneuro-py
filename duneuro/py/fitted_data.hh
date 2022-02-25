#ifndef DUNEURO_PY_FITTED_DATA_HH
#define DUNEURO_PY_FITTED_DATA_HH

#include <dune/python/pybind11/numpy.h>
#include <dune/python/pybind11/pybind11.h>

#include <duneuro/common/fitted_driver_data.hh>

namespace duneuro
{
  template <int dim>
  static void extractFittedDataFromMainDict(pybind11::dict d, duneuro::FittedDriverData<dim>& data)
  {
    if (d.contains("volume_conductor")) {
      auto volume_conductor_dict = d["volume_conductor"].cast<pybind11::dict>();
      if (volume_conductor_dict.contains("grid") && volume_conductor_dict.contains("tensors")) {
        auto grid_dict = volume_conductor_dict["grid"].cast<pybind11::dict>();
        auto tensor_dict = volume_conductor_dict["tensors"].cast<pybind11::dict>();
        if (grid_dict.contains("nodes") && grid_dict.contains("elements")) {
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
        }
        if (tensor_dict.contains("labels")) {
          data.labels = tensor_dict["labels"].cast<std::vector<std::size_t>>();
        }
        if (tensor_dict.contains("conductivities")) {
          data.conductivities = tensor_dict["conductivities"].cast<std::vector<double>>();
        }
        if (tensor_dict.contains("tensors")) {
          for (const auto& t : tensor_dict["tensors"]) {
            auto arr = t.cast<pybind11::array_t<double, pybind11::array::c_style
                                                            | pybind11::array::forcecast>>();
            if (arr.ndim() != 2) {
              DUNE_THROW(Dune::Exception, "a tensor has to be a two dimensional matrix");
            }
            if (arr.shape(0) != dim || arr.shape(1) != dim) {
              DUNE_THROW(Dune::Exception, "tensor has to be a " << dim << "x" << dim << " matrix");
            }
            Dune::FieldMatrix<double, dim, dim> m;
            for (unsigned int i = 0; i < dim; ++i) {
              for (unsigned int j = 0; j < dim; ++j) {
                m[i][j] = *arr.data(i, j);
              }
            }
            data.tensors.push_back(m);
          }
        }
      }
    }
  }
}

#endif // DUNEURO_PY_FITTED_DATA_HH
