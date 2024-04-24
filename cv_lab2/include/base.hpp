#ifndef BASE_HPP
#define BASE_HPP

#include <array>
#include <iostream>
#include <vector>
#include <cassert>
#include <cstring>
#include <memory>

typedef int vec2_t[2];
constexpr vec2_t zeros_default = {0, 0}; 
constexpr vec2_t ones_default = {1, 1}; 
struct Shape {
    Shape() {};
    Shape(std::initializer_list<int> list) {
        assert(list.size() <= 3 && "Tensor shape should has less than 3 dims");
        ndims = list.size();
        shape = std::vector(list);
        total_elements = 1;
        for (auto& dim : shape) { total_elements *= dim; }
    }
    int operator()(int i) { return shape[i]; }

    int32_t ndims;
    int32_t total_elements=0;
    std::vector<int> shape;
};


template <typename _Dt>
class Tensor {
public:
    Tensor() {};
    ~Tensor() {};
    Tensor(Shape shape_) : tensor_shape(shape_){
        ndims = tensor_shape.ndims;
        data.resize(get_total_elements());
    }

    Tensor(Shape shape_, _Dt* data_ptr) : tensor_shape(shape_) {
        ndims = tensor_shape.ndims;
        data = std::vector<_Dt>(data_ptr, data_ptr + get_total_elements());
    }

    Tensor(std::vector<_Dt> data_) {
        tensor_shape = Shape({data.size});
        ndims = tensor_shape.ndims;
        data = std::vector<_Dt>(data_.data(), data_.data() + get_total_elements());
    }

    Tensor(Shape shape_, std::vector<_Dt> data_) {
        tensor_shape = shape_;
        ndims = tensor_shape.ndims;
        assert(tensor_shape.total_elements == data_.size());
        data = std::vector<_Dt>(data_.data(), data_.data() + get_total_elements());
    }

    void reshape(Shape shape_) {
        assert(shape_.total_elements == tensor_shape.total_elements);
        tensor_shape = shape_; // probably copy ???
        ndims = tensor_shape.ndims;
    }

    void fit(Shape shape_) {
        *this = Tensor(shape_);
    }

    void fit(Shape shape_, _Dt* data_) {
        this = Tensor(shape_, data_);
    }

    _Dt& operator()(int i, int j=-1, int k=-1)
    {
        int c_idx = 0, h_idx = 0, w_idx = 0;
        if (ndims == 3) {
            c_idx = i;
            h_idx = j;
            w_idx = k;
            return data[shape(1) * shape(2) * c_idx +
                                   shape(2) * h_idx +
                                              w_idx];
        } else if (ndims == 2) {
            h_idx = i;
            w_idx = j;
            return data[shape(1) * h_idx + w_idx];
        } else {
            w_idx = i;
            return data[w_idx];
        }
    }

    const _Dt& operator()(int i, int j=-1, int k=-1) const {
        int c_idx = 0, h_idx = 0, w_idx = 0;
        if (ndims == 3) {
            c_idx = i;
            h_idx = j;
            w_idx = k;
            return data[shape(1) * shape(2) * c_idx +
                                   shape(2) * h_idx +
                                              w_idx];
        } else if (ndims == 2) {
            h_idx = i;
            w_idx = j;
            return data[shape(1) * h_idx + w_idx];
        } else {
            w_idx = i;
            return data[w_idx];
        }
    }

    const _Dt* get_data() const {
        return data.data();
    }

    _Dt* get_data() {
        return data.data();
    }

    int shape(int i) const {
        assert(i < ndims && i >= 0);
        return tensor_shape.shape[i];
    }

    Shape get_shape() const { return tensor_shape; }
    int32_t get_total_elements() const { return tensor_shape.total_elements; }
    int32_t get_ndims() const { return tensor_shape.ndims; }

private:
    int ndims;
    Shape tensor_shape;
    std::vector<_Dt> data;
};


template<typename _Dt>
std::ostream& operator<<(std::ostream& os, Tensor<_Dt> tensor) {
    int size = tensor.get_shape()(tensor.get_ndims() - 1);
    std::vector<int> dims_sizes(tensor.get_ndims() - 1);
    for (int i = tensor.get_ndims() - 2; i >= 0; --i) {
        dims_sizes[i] = size;
        size *= tensor.get_shape()(i);
    }

    for (int i = 0; i < tensor.get_total_elements(); ++i) {
        for (auto& dim_size : dims_sizes) { 
            if (i != 0 && i % dim_size == 0) os << std::endl;
        }
        os << tensor.get_data()[i] << ",\t";
    }
    std::cout << std::endl;
    return os;
}

template<typename _Dt>
Tensor<_Dt> operator*(Tensor<_Dt> a, Tensor<_Dt> b) {
    assert(a.get_ndims() == 2 && b.get_ndims() == 2);
    assert(a.shape(1) == b.shape(0));

    Tensor<_Dt> res(Shape({a.shape(0), b.shape(1)}));

    for (int row1_idx = 0; row1_idx < a.shape(0); row1_idx++) {
        for (int col2_idx = 0; col2_idx < b.shape(1); col2_idx++) {
            res(row1_idx, col2_idx) = (_Dt)0;
            for (int col1_idx = 0; col1_idx < a.shape(1); col1_idx++) {
                res(row1_idx, col2_idx) += a(row1_idx, col1_idx) * b(col1_idx, col2_idx);
            }
        }
    }

    return res;
}

template<typename _Dt>
bool operator==(Tensor<_Dt> a, Tensor<_Dt> b) {
    if (a.get_ndims() != b.get_ndims())
        return false;
    for (int i = 0; i < a.get_ndims(); ++i) {
        if (a.shape(i) != b.shape(i))
            return false;
    }
    _Dt* a_ptr = a.get_data();
    _Dt* b_ptr = b.get_data();
    for (int i = 0; i < a.get_total_elements(); ++i) {
        if (a_ptr[i] != b_ptr[i])
            return false;
    }

    return true;
}

std::ostream& operator<<(std::ostream& os, Shape shape) {
    for (int i = 0; i < shape.ndims; ++i) {
        os << shape.shape[i] << " ";
    }
    std::cout << std::endl;

    return os;
}
#endif // BASE_HPP
