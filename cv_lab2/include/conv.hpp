#ifndef CONV_HPP
#define CONV_HPP

#include "base.hpp"

template <typename _Dt>
void conv(const Tensor<_Dt>& inp_,
          const std::vector<Tensor<_Dt>>& filters,
                Tensor<_Dt>& out,
          const vec2_t str=ones_default,
          const vec2_t pad=zeros_default);

template <typename _Dt>
void conv_im2col(const Tensor<_Dt>& inp_,
                 const std::vector<Tensor<_Dt>>& filters,
                      Tensor<_Dt>& out,
                const vec2_t str=ones_default,
                const vec2_t pad=zeros_default);
#endif // CONV_HPP
