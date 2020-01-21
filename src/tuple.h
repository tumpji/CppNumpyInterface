#if not defined TUPLE_INCLUDED
#define TUPLE_INCLUDED

#include <stdexcept>

#include "python3.6m/Python.h"

#include "utils/string.h"


//std::enable_if_t<(sizeof...(Args) > 0), bool> = true>
// changes <type,...> to encoding of types by string
// template<typename A1, typename... Args, std::enable_if_t<(sizeof...(Args) > 0)>* = nullptr>

        //std::enable_if_t<(sizeof...(Args) > 0)>* = nullptr>
namespace {


    // I dont have translation...
    template<typename A1>
    constexpr auto generate_string_based_on_template() {
        X_ASSERT(false); // TODO: add other staticLiterals if you want to have in tuple something else
    }
    
    template<>
    constexpr auto generate_string_based_on_template<PyObject*>() {
        return staticLiteral("O");
    }
    template<>
    constexpr auto generate_string_based_on_template<int>(){
        return staticLiteral("i");
    }
    template<>
    constexpr auto generate_string_based_on_template<long int>(){
        return staticLiteral("l");
    }
    template<>
    constexpr auto generate_string_based_on_template<double>(){
        return staticLiteral("d");
    }
    template<>
    constexpr auto generate_string_based_on_template<float>(){
        return staticLiteral("f");
    }
    template<>
    constexpr auto generate_string_based_on_template<char*>(){
        return staticLiteral("s");
    }

    template<typename A1, typename... Args,
        std::enable_if_t<(sizeof...(Args) > 0)>* = nullptr
        >
    constexpr auto generate_string_based_on_template() {
        return generate_string_based_on_template<A1>() + generate_string_based_on_template<Args...>();
    }

    // TODO: add other staticLiterals if you want to have in tuple something else
}

class PyTuple {
public:
    template<typename... Args>
    PyTuple(Args... args) {
      object = Py_BuildValue( (staticLiteral("(") + generate_string_based_on_template<Args...>() + staticLiteral(")")).cstring(), args...);

      if (object == nullptr) {
          throw std::runtime_error("Cannot create tuple");
      }
    }

    ~PyTuple() {
        Py_DECREF(object);
        object = nullptr;
    }

    PyObject* pass_to_python() {
        Py_INCREF(object);
        return object;
    }
private:
    PyObject* object;
};

#endif // TUPLE_INCLUDED
