#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/tucker_layer.hpp"

namespace caffe {

template <typename Dtype>
void TuckerLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  TuckerParameter tucker = this->layer_param_.tucker_param();
  mse = tucker.mse();
  Obits = tucker.test();
}

template <typename Dtype>
void TuckerLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  top[0]->ReshapeLike(*bottom[0]);
  
  num_ = bottom[0]->num();
  ch_ = bottom[0]->channels();
  h_ = bottom[0]->height();
  w_ = bottom[0]->width();

}

template <typename Dtype>
void TuckerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

    Dtype* top_data = top[0]->mutable_cpu_data();
    const Dtype* bottom_data = bottom[0]->cpu_data();

    // LOG(INFO) << bottom[0]->shape(0) << " " << bottom[0]->shape(1) << " " << bottom[0]->shape(2)<< " " << bottom[0]->shape(3);
    // LOG(INFO) << this->num_;

    s = vector<uint32_t> (3);
    s[0] = ch_;
    s[1] = h_;
    s[2] = w_;
    n = 3;

    sprod = vector<size_t> (n+1);
    sprod[0] = 1;
    sprod[1] = ch_;
    sprod[2] = ch_*h_;
    sprod[3] = ch_*h_*w_;
    
    for (int32_t batch = 0; batch < this->num_; batch++){
      core.clear();
      core_t.clear();
      minimums.clear();
      maximums.clear();  
      r.clear();
      rprod.clear();
      sorting.clear();
      chunk_ids.clear();
      Us_q.clear();
      Us_t.clear();
      Us.clear();

      size_t size = sprod[n];
      vector<double> core(size); 
      vector<MatrixXd> Us(n);
      vector< pair<double,size_t> > sorting(size);
      
      for (int32_t i = 0; i < size; i++){
        core[i] = bottom_data[i + (batch*sprod[n])];      
        datamin = min(datamin, core[i]); 
        datamax = max(datamax, core[i]);
        datanorm += core[i] * core[i];   
      }

      datanorm = sqrt(datanorm);
      sse = pow((datamax - datamin) / (2 * (pow(10, mse / 20))), 2) * size;
      double lim = sse / size;

      // ****************************** Tensor Compress *************************** //
      hosvd_compress(core, Us);

      for (int32_t i = 0; i < size; ++i){
        sorting[i] = pair < double, size_t > (abs(core[i]), i);
      }
      sort(sorting.begin(), sorting.end());

      // ****************************** Quantization **************************** //
      size_t adder = 1;
      uint8_t q = 0;
      size_t left = 0;
      size_t old_right = left;
      size_t right = left;
      size_t hbits = 0;
      vector<uint8_t> chunk_ids(size, 0);
      uint8_t chunk_num = 1;
      vector<double> minimums(32);
      vector<double> maximums(32);
      vector< vector<uint8_t> > Us_q(n);
      for (int32_t i = 0; i < n; ++i){
        Us_q[i] = vector<uint8_t> (s[i], 0);
      }

      while (left < size) {
        while (left < size and q < 32) {
          right = min(size, old_right + adder);
          double chunk_min = sorting[left].first;
          double chunk_max = sorting[right - 1].first;
          double sse = 0;
          if (right > left + 1) {
            if (q > 0) {
              for (int32_t i = left; i < right; ++i) {
                uint64_t quant = roundl((sorting[i].first - chunk_min) * ((1UL << q) - 1.) / (chunk_max - chunk_min));
                double dequant = quant * (chunk_max - chunk_min) / ((1UL << q) - 1.) + chunk_min;
                sse += (sorting[i].first - dequant) * (sorting[i].first - dequant);
              }
            } 
            else {
              for (int32_t i = left; i < right; ++i)
                sse += (sorting[i].first - chunk_min) * (sorting[i].first - chunk_min);
            }
          }
          double mse = sse / (right - left);
          if (mse >= 0.9 * lim or right == size) {
            if (mse >= lim) {
              if (adder > 1) {
                adder = ceil(adder / 4.);
                continue;
              } 
              else {
                right = old_right;
                break;
              }
            } 
            else
              break;
          } 
          else {
            old_right = right;
            adder *= 2;
          }
        }
        if (q == 32)
          right = size;

        size_t chunk_size = (right - left);
        double chunk_min = sorting[left].first;
        double chunk_max = sorting[right - 1].first;

        if (q > 0 and q < 32) {
         for (int32_t i = left; i < right; ++i) {
          size_t to_write = 0;
          if (chunk_size > 1)
             to_write = min(((1UL << q) - 1), (uint64_t) roundl((sorting[i].first - chunk_min) / (chunk_max - chunk_min) * ((1UL << q) - 1)));
          if (core[sorting[i].second] < 0)
              to_write |= 1UL << q;
          core[sorting[i].second] = to_write;
         }
        }
        
        for (size_t i = left; i < right; ++i) {
          size_t index = sorting[i].second;
          chunk_ids[index] = chunk_num;
          for (uint8_t dim = 0; dim < n; ++dim) {
            size_t coord = index % sprod[dim+1] / sprod[dim];
            Us_q[dim][coord] = max(Us_q[dim][coord], q);
          }
        }

        if (Obits){
	        vector<size_t> rle;
	        bool current_bit = false;
	        bool last_bit = false;
	        size_t counter = 0;
	        for (size_t i = left; i < right; ++i) {
	            if (chunk_ids[i] == 0)
	                current_bit = false;
	            else if (chunk_ids[i] == chunk_num)
	                current_bit = true;
	            else
	                continue;
	            if (current_bit == last_bit)
	                counter++;
	            else {
	                rle.push_back(counter);
	                counter = 1;
	                last_bit = current_bit;
	            }
	        }
	        rle.push_back(counter);
	        size_t n_hbits = codebook(rle);
            hbits += n_hbits;
	    }

        maximums[q] = chunk_max;
        minimums[q] = chunk_min;
        q++;
        left = right;
        old_right = left;
        chunk_num++;
      }
      // **************** Tensor rank: Size of Core and factor matrices *************** //
      r = vector<uint32_t> (n);
      rprod = vector<size_t> (n+1);
      rprod[0] = 1;
      for (int32_t i = 0; i < n; ++i) {
        for (int32_t j = 0; j < s[i]; ++j){
          if (Us_q[i][j])
            r[i] = j+1;
        }
        rprod[i+1] = rprod[i]*r[i];
      }

      vector<double> core_t(rprod[n]);                                // core
      size_t j = 0;  
      size_t bits = 0;                                                  
      for (size_t i = 0; i < size; ++i) {
        uint8_t q = chunk_ids[i]-1;
        if (q > 0){
            core_t[j] = core[i];
            bits += q;
            j++;
          }
      }

      vector<MatrixXd> Us_t(n);                                       // factor matrices
      size_t ubit = 0;
      for (uint8_t i = 0; i < n; ++i){
        size_t ubits = factor(Us[i].leftCols(r[i]), Us_q[i], Us_t[i]);
        ubit += ubits;
      }

      // bits of Us_q
      if (Obits){
        size_t uqbit = 0;
        for (uint8_t i = 0; i < n; ++i){
           uqbit += r[i]*sizeof(uint8_t)*8;
        }
        size_t total = bits + uqbit + ubit + hbits;
        LOG(INFO) << total << " " << h_*8*w_*8*3*8;
        LOG(INFO) << r[0] << " " << r[1] << " " << r[2];
      }

      // ****************************** De-Quantization ************************** //
      for (size_t i = 0; i < rprod[n]; ++i) {                        // i marks where to write in the new rank-reduced core
          uint8_t q = chunk_ids[i]-1;
          if (q > 0) {
              double chunk_min = minimums[q];
              double chunk_max = maximums[q];
              uint64_t quant = core[i];

              uint8_t sign = (quant >> q) & 1UL;                    // Read the sign bit
              quant &= ~(1UL << q);                                 // Put the sign bit to zero
              double dequant;
              dequant = quant / ((1UL << q) - 1.) * (chunk_max - chunk_min) + chunk_min;
              core_t[i] = -(sign * 2 - 1) * dequant;  
          } else
              core_t[i] = 0;
      }

      // ****************************** Tensor Deompress ************************** //
      hosvd_decompress(core_t, Us_t);

      for (int i = 0; i < size; i++){
       top_data[i + (batch*sprod[n])] = core_t[i]; 
      }  
   }
}
  

template <typename Dtype>
void TuckerLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* top_diff = top[0]->cpu_diff();

  for (int i = 0; i < bottom[0]->count(); i++){
    bottom_diff[i] = top_diff[i];
  }
}

#ifdef CPU_ONLY
    STUB_GPU(TuckerLayer);
#endif

    INSTANTIATE_CLASS(TuckerLayer);
    REGISTER_LAYER_CLASS(Tucker);

}  // namespace caffe
