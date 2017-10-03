#include <iostream>
#include <stdlib.h>
#include <string.h>
#include "feature_group.h"
#include "v_arary.h"


struct ngram{
    size_t n;
    v_array<uint64_t> indexes;
    bool used;

    ngram(size_t num, size_t start, const v_array<uint64_t>& index){
        n = num;
        used = false;
        indexes = v_init<uint64_t>();
        for (size_t t = 0; t < n; t ++)
            indexes.push_back(index[start+t]);
    }
};

inline bool two_ngrams_equal(const ngram& ngram1, const ngram& ngram2)
{
    if (ngram1.n != ngram2.n)
        return false;
    
    for (size_t t = 0 ; t < ngram1.n; t++){
        if (ngram1.indexes[t] != ngram2.indexes[t])
            return false;
    }
    return true;
}
inline void create_ngram(size_t n, const v_array<uint64_t>& indicies, v_array<ngram>& ngrams)
{
    size_t len = indicies.size();
    ngrams = v_init<ngram>();
    for (size_t t = 0; t <= len - n; t ++){
        ngram single_ngram = ngram(n, t, indicies);
        ngrams.push_back(single_ngram);
    }
}

inline uint32_t num_ngram_in_given(const ngram& n_gram, const v_array<ngram>& given_ngrams, bool self = true)
{
    uint32_t com_num = 0;
    size_t len = ngrams_self.size();

    for (size_t t = 0; t < len; t++){
        if (self == false){
            if(two_ngrams_equal(n_gram, given_ngrams[t]) == true){
                com_num ++;
            }
        }
        else if (self == true){
            if (given_ngrams[t].used == false){
                if (two_ngrams_equal(n_gram, given_ngrams[t]) == true){
                    com_num ++;
                    given_ngrams[t].used = true;
                }
            }
        }
    }
    return com_num;
}

inline uint32_t clip_count(const ngram& n_gram, const v_array<ngram>& candiate_ngrams, const v_array<ngram>& reference_ngrams)
{
    uint32_t com_num_self = num_ngram_in_given(n_gram, candiate_ngrams, true);
    uint32_t com_num_refe = num_ngram_in_given(n_gram, reference_ngrams,false);

    return (com_num_self >= com_num_refe)? com_num_refe : com_num_self;
}

inline float ngram_precision(const v_array<ngram>& candidate_ngrams, const v_array<ngram>& reference_ngrams)
{
    uint32_t dist_total_counts = 0;
    size_t len_can = candiate_ngrams.size();
    for (size_t t = 0; t < len_can; t++){
        ngram& curr_gram = candiate_ngrams[t];
        if (curr_gram.used == false){
            dist_total_counts += clip_count(curr_gram, candiate_ngrams, reference_ngrams);
        } 
    }
    return dist_total_counts*1./len_can;
}

inline float bleu(const v_array<uint64_t>& candidate, const v_array<uint64_t>& reference)
{
    float len_ratio = candiate.size()*1./reference.size();
    float brevity = (len_ratio <= 1.) ? len_ratio : 1.;
    float precision = 1.;
    for (int i = 1; i <= 4; i++){
        v_array<ngram> ngram_candidate = v_init<ngram>();
        v_array<ngram> ngram_reference= v_init<ngram>();
        create_ngram(i, candidate, ngram_candidate);
        create_ngram(i, reference, ngram_reference);
        float p_i = ngram_precision(ngram_candidate, ngram_reference);
        precision*=p_i;
    }
    float bleu = brevity * pow(precision, 0.25);
    return bleu;
}



