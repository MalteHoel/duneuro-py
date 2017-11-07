#include <duneuro/py/parameter_tree.h>

namespace duneuro
{
  pybind11::dict toPyDict(const Dune::ParameterTree& tree)
  {
    pybind11::dict out;
    for (const auto& key : tree.getValueKeys()) {
      out[key.c_str()] = pybind11::str(tree[key]);
    }
    for (const auto& subkey : tree.getSubKeys()) {
      out[subkey.c_str()] = toPyDict(tree.sub(subkey));
    }
    return out;
  }

  std::string toString(pybind11::handle handle)
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
        str << toString(*it);
      }
      return str.str();
    }
    DUNE_THROW(StringConversionException, "type \"" << type << "\" not supported");
  }

  std::map<std::string, std::string> toStringMap(pybind11::dict dict)
  {
    std::map<std::string, std::string> map;
    for (const auto& item : dict) {
      std::string type = item.second.get_type().attr("__name__").cast<std::string>();
      std::string key = toString(item.first);
      if (type == "dict") {
        auto sub = toStringMap(item.second.cast<pybind11::dict>());
        for (const auto& k : sub) {
          map[key + "." + k.first] = k.second;
        }
      } else {
        try {
          map[key] = toString(item.second);
        } catch (StringConversionException& ex) {
          // ignore the entry. will be triggered for numpy arrays, which we do not want to be
          // converted to string
        }
      }
    }
    return map;
  }

  Dune::ParameterTree toParameterTree(pybind11::dict dict)
  {
    Dune::ParameterTree tree;
    auto map = toStringMap(dict);
    for (const auto& k : map) {
      tree[k.first] = k.second;
    }
    return tree;
  }
}
