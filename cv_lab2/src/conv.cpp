#include "conv.hpp"

template <typename _Dt>
void checks(const Tensor<_Dt>& inp_,
            const std::vector<Tensor<_Dt>>& filters,
                  Tensor<_Dt>& out,
            const vec2_t str,
            const vec2_t pad) {
    assert(inp_.get_ndims() == 3);

    int n_filters = filters.size();
    assert(n_filters > 0);

    int channels = inp_.shape(0);
    for (int i = 0; i < n_filters; ++i) {
        assert(channels == filters[i].shape(0));
        assert(filters[i].get_ndims() == 3);
        for (auto j = 1; j < filters[0].get_ndims(); ++j) {
            assert(filters[i].shape(j) == filters[0].shape(j) && filters[i].shape(j) <= inp_.shape(j));
        }
    }
}

template <typename _Dt>
Tensor<_Dt> add_pad(const Tensor<_Dt>& inp_, const vec2_t pad) {
    int channels    = inp_.shape(1);
    int height = inp_.shape(1);
    int width  = inp_.shape(2);  
  
    Tensor<_Dt> inp(Shape({channels, height + 2*pad[0], width + 2*pad[1]}));
    if (pad != zeros_default) {
        std::fill(inp.get_data(), inp.get_data() + inp.get_total_elements(), 0);
        for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    inp(c, h + pad[0], w + pad[1]) = inp_(c, h, w);
                }
            }
        }
    } else {
        inp = inp_;
    }
    return inp;
}

template <typename _Dt>
void conv(const Tensor<_Dt>& inp_,
          const std::vector<Tensor<_Dt>>& filters,
                Tensor<_Dt>& out,
          const vec2_t str,
          const vec2_t pad) {

    checks(inp_, filters, out, str, pad);

    Tensor<_Dt> inp = add_pad(inp_, pad);

    int channels   = inp_.shape(0);
    int n_filters  = filters.size();
    int inp_height = inp.shape(1);
    int inp_width  = inp.shape(2);

    // fit output shape
    int fil_height = filters[0].shape(1);
    int fil_width  = filters[0].shape(2);

    assert(inp_height >= fil_height);
    assert(inp_width >= fil_width);
    
    int out_height = (inp_height - 2 * pad[0] - (fil_height - 1) - 1) / str[0] + 1;
    int out_width =  (inp_width  - 2 * pad[1] - (fil_width  - 1) - 1) / str[1] + 1;
    out.fit(Shape({n_filters, out_height, out_width}));

    // simple conv implementation
    for (auto f = 0; f < n_filters; ++f) {
        for (auto out_h = 0; out_h < out_height; ++out_h) {
            for (auto out_w = 0; out_w < out_width; ++out_w) {
                for (auto c = 0; c < channels; ++c) {
                    for (auto fil_h = 0; fil_h < fil_height; ++fil_h) {
                        for (auto fil_w = 0; fil_w < fil_width; ++fil_w) {
                            out(f, out_h, out_w) += filters[f](c, fil_h, fil_w) *
                                                                inp(c, str[0] * out_h + fil_h, str[1] * out_w + fil_w);
                        }
                    }
                }
            }
        }
    }
}

template <typename _Dt>
void conv_im2col(const Tensor<_Dt>& inp_,
                 const std::vector<Tensor<_Dt>>& filters,
                      Tensor<_Dt>& out,
                const vec2_t str,
                const vec2_t pad) {
        checks(inp_, filters, out, str, pad);

    Tensor<_Dt> inp = add_pad(inp_, pad);

    int channels   = inp_.shape(0);
    int n_filters  = filters.size();
    int inp_height = inp.shape(1);
    int inp_width  = inp.shape(2);

    // fit output shape
    int fil_height = filters[0].shape(1);
    int fil_width  = filters[0].shape(2);

    assert(inp_height >= fil_height);
    assert(inp_width >= fil_width);
    
    int out_height = (inp_height - 2 * pad[0] - (fil_height - 1) - 1) / str[0] + 1;
    int out_width =  (inp_width  - 2 * pad[1] - (fil_width  - 1) - 1) / str[1] + 1;
    out.fit(Shape({n_filters, out_height, out_width}));

    // create 2d tensor for filters
    Tensor<_Dt> filters2d(Shape({n_filters, channels * fil_height * fil_width}));
    for (int f_n = 0; f_n < n_filters; ++f_n) {
        for(int c = 0; c < channels; ++c) {
            for (int f_h = 0; f_h < fil_height; ++f_h) {
                for (int f_w = 0; f_w < fil_width; ++f_w) {
                    filters2d(f_n, c * fil_height * fil_width + f_h * fil_width + f_w) =
                    filters[f_n](c, f_h, f_w);
                }
            }
        }
    }

    // create 2d tensor for inp
    Tensor<_Dt> inp2d(Shape({channels * fil_height * fil_width, out_height * out_width}));
    int col_index = 0;
    for (int i = 0; i < out_height; ++i) {
        for (int j = 0; j < out_width; ++j) {
            for (int ci = 0; ci < channels; ++ci) {
                for (int ki = 0; ki < fil_height; ++ki) {
                    for (int kj = 0; kj < fil_width; ++kj) {
                        int row = i * str[0] + ki;
                        int col = j * str[1] + kj;
                        if (row < 0 || row >= inp_height || col < 0 || col >= inp_width) {
                            inp2d(col_index, i * out_width + j) = 0.0;
                        } else {
                            inp2d(ci * fil_height * fil_width + col_index, i * out_width + j) = inp(ci, row, col);
                        }
                        col_index ++;
                    }
                }
                col_index = 0;
            }
        }
    }

    // get 2d tensor with res
    Tensor<_Dt> out2d = filters2d * inp2d;

    // create 3d tensor for res
    for (int f_n = 0; f_n < n_filters; ++f_n) {
        for (int out_h = 0; out_h < out_height; ++out_h) {
            for (int out_w = 0; out_w < out_width; ++out_w) {
                out(f_n, out_h, out_w) = out2d(f_n, out_h * out_height + out_w);
            }
        }
    }
}


template void conv<int>( const Tensor<int>&,
                        const std::vector<Tensor<int>>&,
                              Tensor<int>&,
                        const vec2_t,
                        const vec2_t);
template void conv<float>(  const Tensor<float>&,
                        const std::vector<Tensor<float>>&,
                              Tensor<float>&,
                        const vec2_t,
                        const vec2_t);
template void conv_im2col<int>(  const Tensor<int>&,
                        const std::vector<Tensor<int>>&,
                              Tensor<int>&,
                        const vec2_t,
                        const vec2_t);
template void conv_im2col<float>( const Tensor<float>&,
                        const std::vector<Tensor<float>>&,
                              Tensor<float>&,
                        const vec2_t,
                        const vec2_t);
