#ifndef CAFFE_TUCKER_LAYER_HPP_
#define CAFFE_TUCKER_LAYER_HPP_

#include <vector>
#include <unistd.h>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/Tucker/Define.hpp"
#include "caffe/util/Tucker/hosvd.hpp"
#include "caffe/util/Tucker/Codebook.hpp"

namespace caffe {

template <typename Dtype>

class TuckerLayer : public Layer<Dtype> {
   public:
    explicit TuckerLayer(const LayerParameter& param)
         : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
         const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
         const vector<Blob<Dtype>*>& top);
    virtual inline const char* type() const { return "Tucker"; }
    virtual inline int ExactNumBottomBlobs() const { return 1; }
    virtual inline int ExactNumTopBlobs() const { return 1; }
   
   protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
  //  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  //      const vector<Blob<Dtype>*>& top);
 
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
/*          
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
*/

    int num_, ch_, h_, w_;
    Dtype mse;
    bool Obits;
	};
} 

#endif //  CAFFE_TUCKER_LAYER_HPP_
