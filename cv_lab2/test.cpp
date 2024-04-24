#include <iostream>
#include "base.hpp"
#include "conv.hpp"

int main() {
    Tensor<int> test_tensor(Shape({2, 8, 8}), std::vector<int>({55719,  56070,  56421,  56772,  57123,  57474,  57825,  58176,
                                                                59229,  59580,  59931,  60282,  60633,  60984,  61335,  61686,
                                                                62739,  63090,  63441,  63792,  64143,  64494,  64845,  65196,
                                                                66249,  66600,  66951,  67302,  67653,  68004,  68355,  68706,
                                                                69759,  70110,  70461,  70812,  71163,  71514,  71865,  72216,
                                                                73269,  73620,  73971,  74322,  74673,  75024,  75375,  75726,
                                                                76779,  77130,  77481,  77832,  78183,  78534,  78885,  79236,
                                                                80289,  80640,  80991,  81342,  81693,  82044,  82395,  82746,

                                                                58716,  59094,  59472,  59850,  60228,  60606,  60984,  61362,
                                                                62496,  62874,  63252,  63630,  64008,  64386,  64764,  65142,
                                                                66276,  66654,  67032,  67410,  67788,  68166,  68544,  68922,
                                                                70056,  70434,  70812,  71190,  71568,  71946,  72324,  72702,
                                                                73836,  74214,  74592,  74970,  75348,  75726,  76104,  76482,
                                                                77616,  77994,  78372,  78750,  79128,  79506,  79884,  80262,
                                                                81396,  81774,  82152,  82530,  82908,  83286,  83664,  84042,
                                                                85176,  85554,  85932,  86310,  86688,  87066,  87444,  87822}));
    // data preparation
    Tensor<int> tensor(Shape({3, 10, 10}));
    for (auto i = 0; i < tensor.get_total_elements(); ++i) {
        tensor.get_data()[i] = i;
    }

    Tensor<int> filter1(Shape({3, 3, 3}));
    for (auto i = 0; i < filter1.get_total_elements(); ++i) {
        filter1.get_data()[i] = i;
    }

    Tensor<int> filter2(Shape({3, 3, 3}));
    for (auto i = 0; i < filter2.get_total_elements(); ++i) {
        filter2.get_data()[i] = i+1;
    }

    //------------- testing conv--------------
    Tensor<int> out;
    std::vector<Tensor<int>> filters({filter1, filter2});
    conv(tensor, filters, out);
    
    // comparison with test tensor
    std::cout << "Tensors are equal: " << (out == test_tensor) << std::endl;

    //--------- testing im2col conv----------
    Tensor<int> out_im2col;
    vec2_t stride = {2, 2};
    vec2_t pad = {2, 2};
    conv(tensor, filters, out, stride, pad);
    conv_im2col(tensor, filters, out_im2col, stride, pad);

    std::cout << "Tensors are equal: " << (out == out_im2col) << std::endl;
}