#ifndef DUNEURO_PY_PARAMETER_TREE_H
#define DUNEURO_PY_PARAMETER_TREE_H

#include <map>

#include <pybind11/pybind11.h>

#include <dune/common/parametertree.hh>

namespace duneuro
{
  class StringConversionException : public Dune::Exception {};

  pybind11::dict toPyDict(const Dune::ParameterTree& tree);
  std::string toString(pybind11::handle handle);
  std::map<std::string, std::string> toStringMap(pybind11::dict dict);
  Dune::ParameterTree toParameterTree(pybind11::dict dict);
}

#endif // DUNEURO_PY_PARAMETER_TREE_H
