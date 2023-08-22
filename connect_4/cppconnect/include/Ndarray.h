#ifndef C___CONNECT_4_NDARRAY_H
#define C___CONNECT_4_NDARRAY_H

#include <cppflow/cppflow.h>
#include <vector>
#include <cstdint>
#include <utility>

template<class T>
class Ndarray {
public:
    explicit Ndarray(std::vector<int64_t>&& s) : shape{std::move(s)}
    {
        set_dimensions();
        buffer.resize(dimensions[0]*shape[0]);
    }

    Ndarray(std::vector<T>&& data, std::vector<int64_t>&& s) : buffer{std::move(data)}, shape{std::move(s)}
    {
        set_dimensions();
    }

    T& operator[](const std::vector<int>& indices)
    {
        int index{0};
        for (int i{0}; i<indices.size(); ++i) {
            if (indices[i]<0 || indices[i]>=shape[i]) {
                return index_error;
            }
            index += indices[i]*dimensions[i];
        }
        return buffer[index];
    }

    T operator[](const std::vector<int>& indices) const
    {
        int index{0};
        for (int i{0}; i<indices.size(); ++i) {
            if (indices[i]<0 || indices[i]>=shape[i]) {
                return index_error;
            }
            index += indices[i]*dimensions[i];
        }
        return buffer[index];
    }

    [[nodiscard]] cppflow::tensor get_tensor() const
    {
        return cppflow::tensor{buffer,shape};
    }

private:
    void set_dimensions()
    {
        dimensions.resize(shape.size());
        for (int i{static_cast<int>(shape.size())-1}; i>=0; --i) {
            dimensions[i] = (i==(shape.size()-1)) ? 1 : shape[i+1]*dimensions[i+1];
        }
    }

    std::vector<T> buffer;
    const std::vector<int64_t> shape;
    std::vector<int> dimensions;
    T index_error{0};
};

#endif //C___CONNECT_4_NDARRAY_H
