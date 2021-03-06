#include <algorithm>
#include <cmath>
#include <cstdio>
#include <float.h>
#include <time.h>
#include <sstream>

#include "reductions.h"
#include "rand48.h"
#include "vw.h"


using namespace std;
using namespace LEARNER;

namespace memory_tree_xml_ns
{
    ///////////////////////Helper//////////////////////////////
    //////////////////////////////////////////////////////////
    template<typename T> 
    void pop_at_index(v_array<T>& array, uint32_t index)
    {
        if (index >= array.size()){
            cout<<"ERROR: index is larger than the size"<<endl;
            return;
        }
        if (index == array.size() - 1){
            array.pop();
            return;
        }
        for (size_t i = index+1; i < array.size(); i++){
            array[i-1] = array[i];
        }
        array.pop();
        return;
    }

    inline void copy_example_data(example* dst, const example* src, bool no_multilabel = false, bool no_feat = false)
    { 
        if (!no_multilabel){
            //dst->l.multilabels = src->l.multilabels;
            copy_array(dst->l.multilabels.label_v, src->l.multilabels.label_v);
        }
        else{
            dst->l = src->l;
            dst->l.multi.label = src->l.multi.label;
        }

        copy_array(dst->tag, src->tag);
        dst->example_counter = src->example_counter;
        dst->ft_offset = src->ft_offset;

        if (no_feat == false){
            copy_array(dst->indices, src->indices);
            for (size_t t = 0; t < src->indices.size(); t++)
            //for (namespace_index c : src->indices)
                dst->feature_space[src->indices[t]].deep_copy_from(src->feature_space[src->indices[t]]);
            dst->ft_offset = src->ft_offset;

            dst->num_features = src->num_features;
        }
        dst->partial_prediction = src->partial_prediction;
        if (src->passthrough == nullptr) dst->passthrough = nullptr;
        else
        { 
            dst->passthrough = new features;
            dst->passthrough->deep_copy_from(*src->passthrough);
        }
        dst->loss = src->loss;
        dst->weight = src->weight;
        dst->total_sum_feat_sq = src->total_sum_feat_sq;
        dst->confidence = src->confidence;
        dst->test_only = src->test_only;
        dst->end_pass = src->end_pass;
        dst->sorted = src->sorted;
        dst->in_use = src->in_use;
    }

    inline void free_example(example* ec)
    {
        //ec->l.multilabels.label_v.delete_v();
        ec->tag.delete_v();
        if (ec->passthrough){
            ec->passthrough->delete_v();
        }
        for (int i = 0; i < 256; i++)
            ec->feature_space[i].delete_v();
        ec->indices.delete_v();
        free(ec);
    }

    void remove_repeat_features_in_f(features& f)
    {
        for (size_t i = 0; i < f.indicies.size(); i++){
            if (f.values[i] != -FLT_MAX){
                uint64_t loc = f.indicies[i];
                for (size_t j = i+1; j < f.indicies.size(); j++){
                    if (loc == f.indicies[j]){
                        f.values[i] += f.values[j];
                        f.values[j] = -FLT_MAX;
                    }
                }
            }
        }
        for (size_t i = 0; i < f.indicies.size(); i++){
            if (f.values[i] == -FLT_MAX)
                f.values[i] = 0.0;
        }
    }
    
    void remove_repeat_features_in_ec(example& ec)
    {
        for (auto nc : ec.indices)
            remove_repeat_features_in_f(ec.feature_space[nc]);
    }

    ////Implement kronecker_product between two examples:
    //kronecker_prod at feature level:
    void diag_kronecker_prod_fs(features& f1, features& f2, float& total_sum_feat_sq, float norm_sq1, float norm_sq2)
    {
        v_array<feature_index> tmp_f2_indicies = v_init<feature_index>();
        copy_array(tmp_f2_indicies, f2.indicies);
        for (size_t i1 = 0; i1 < f1.indicies.size(); i1++){
            size_t i2 = 0;
            for (i2 = 0; i2 < f2.indicies.size(); i2++){
                if (f1.indicies[i1] == f2.indicies[i2]){
                    f1.values[i1] = f1.values[i1]*f2.values[i2]/pow(norm_sq1*norm_sq2,0.5) - abs(f1.values[i1]/pow(norm_sq1,0.5) - f2.values[i2]/pow(norm_sq2,0.5));
                    //f1.values[i1] = f1.values[i1]*f2.values[i2] / pow(norm_sq1*norm_sq2, 0.5);
                    total_sum_feat_sq += pow(f1.values[i1],2);
                    tmp_f2_indicies[i2] = 0;
                    break;
                }
            }
            if (i2 == f2.indicies.size()){ //f1's index does not appear in f2, namely value of the index in f2 is zero.
                //f1.values[i1] = 0.0;
                f1.values[i1] = 0.0 - abs(f1.values[i1]/pow(norm_sq1,0.5));
                total_sum_feat_sq += pow(f1.values[i1],2);
            }
        }
        for (size_t i2 = 0; i2 < tmp_f2_indicies.size(); i2++){
            if (tmp_f2_indicies[i2] != 0){
                float value = 0.0 - abs(f2.values[i2]/pow(norm_sq2,0.5));
                f1.push_back(value, f2.indicies[i2]);
                total_sum_feat_sq += pow(value, 2);
            }
        }
        tmp_f2_indicies.delete_v();
    }
    //kronecker_prod at example level:
    void diag_kronecker_product(example& ec1, example& ec2, example& ec)
    {
        //ec <= ec1 X ec2
        copy_example_data(&ec, &ec1, true, false);
        ec.total_sum_feat_sq = 0.0;
        for(namespace_index c : ec.indices){
            for(namespace_index c2 : ec2.indices){
                if (c == c2){
                    diag_kronecker_prod_fs(ec.feature_space[c], ec2.feature_space[c2], ec.total_sum_feat_sq, ec1.total_sum_feat_sq, ec2.total_sum_feat_sq);
                }
            }
        }
    }

    void kronecker_product_f(features& f1, features& f2, features& f, float& total_sq, size_t& num_feat, uint64_t mask, size_t ss)
    {
        for (size_t i = 0; i < f1.indicies.size(); i++){
            size_t j = 0;
            for (j = 0; j < f2.indicies.size(); j++){
                if (f1.indicies[i] == f2.indicies[j]){    // != 0
                    f.push_back(f1.values[i]*f2.values[j], ((f1.indicies[i]+f2.indicies[j])<<ss) & mask);
                    total_sq += pow(f1.values[i]*f2.values[j], 2);
                    num_feat ++;
                    break;
                }
            }
        }
    }

    void kronecker_product(example& ec1, example& ec2, example& ec)
    {
        copy_example_data(&ec, &ec1, true); 
        ec.indices.delete_v();
        //ec.indices.push_back(conditioning_namespace); //134, x86
        //ec.indices.push_back(dictionary_namespace); //135 x87
        unsigned char namespace_1 = 'a';
        unsigned char namespace_2 = 'b';
        ec.indices.push_back(namespace_1); //134, x86
        ec.indices.push_back(namespace_2); //135 x87
        ec.num_features = 0;
        ec.total_sum_feat_sq = 0.0;

        //to do: figure out how to set the indicies correctly (different namespaces may have same index)
        for (auto nc : ec1.indices)
        {
            for (size_t i = 0; i < ec1.feature_space[nc].indicies.size(); i++){
                //ec.feature_space[conditioning_namespace].push_back(ec1.feature_space[nc].values[i], ec1.feature_space[nc].indicies[i]);
                ec.feature_space[namespace_1].push_back(ec1.feature_space[nc].values[i], ec1.feature_space[nc].indicies[i]);
                ec.num_features++;
                ec.total_sum_feat_sq+=pow(ec1.feature_space[nc].values[i],2);
            }
        }
        for (auto nc : ec2.indices)
        {
            for (size_t i = 0; i < ec2.feature_space[nc].indicies.size(); i++){
                //ec.feature_space[dictionary_namespace].push_back(ec2.feature_space[nc].values[i], ec2.feature_space[nc].indicies[i]);
                ec.feature_space[namespace_2].push_back(ec2.feature_space[nc].values[i], ec2.feature_space[nc].indicies[i]);
                ec.num_features++;
                ec.total_sum_feat_sq+=pow(ec2.feature_space[nc].values[i],2);
            }
        }
    }

    ////////////////////////////end of helper/////////////////////////
    //////////////////////////////////////////////////////////////////


    ////////////////////////Implementation of memory_tree///////////////////
    ///////////////////////////////////////////////////////////////////////

    //construct node for tree.
    struct node
    {
        uint32_t parent; //parent index
        int internal;
        //bool internal; //an internal or leaf
        uint32_t depth; //depth.
        uint32_t base_router; //use to index router.
        uint32_t left;  //left child.
        uint32_t right; //right child.

        double nl; //number of examples routed to left.
        double nr; //number of examples routed to right.
        
        v_array<uint32_t> examples_index;

        node () //construct:
        {
            parent = 0;
            internal = 0; //0:not used, 1:internal, -1:leaf 
            //internal = false;
            depth = 0;
            base_router = 0;
            left = 0;
            right = 0;
            nl = 0.001; //initilze to 1, as we need to do nl/nr.
            nr = 0.001; 
            examples_index = v_init<uint32_t>();
        }
    };

    struct score_label
    {
        uint32_t label;
        float score;
        score_label()
        {
            label = 0;
            score = 0;
        }
        score_label(uint32_t l, float s){
            label = l;
            score = s;
        }
    };


    //memory_tree
    struct memory_tree
    {
        vw* all;

        v_array<node> nodes;  //array of nodes.
        v_array<example*> examples; //array of example points
        
        size_t max_leaf_examples; 
        size_t max_nodes;
        size_t max_routers;
        float alpha; //for cpt type of update.
        size_t routers_used;
        int iter;

        size_t max_depth;
        size_t max_ex_in_leaf;


        bool path_id_feat;

        float num_mistakes;
        uint32_t num_ecs;
        uint32_t num_test_ecs;
        uint32_t test_mistakes;
        bool learn_at_leaf;

        bool test_mode;

        int K;

        memory_tree(){
            nodes = v_init<node>();
            examples = v_init<example*>();
            alpha = 0.5;
            routers_used = 0;
            iter = 0;
            num_mistakes = 0.;
            num_ecs = 0;
            num_test_ecs = 0;
            path_id_feat = false;
            test_mode = false;
            max_depth = 0;
            max_ex_in_leaf = 0;
            K = 1;
        }
    };

    float linear_kernel(const flat_example* fec1, const flat_example* fec2)
    { float dotprod = 0;
   
     features& fs_1 = (features&)fec1->fs;
     features& fs_2 = (features&)fec2->fs;
     if (fs_2.indicies.size() == 0)
       return 0.f;
   
     int numint = 0;
     for (size_t idx1 = 0, idx2 = 0; idx1 < fs_1.size() && idx2 < fs_2.size() ; idx1++)
     { uint64_t ec1pos = fs_1.indicies[idx1];
       uint64_t ec2pos = fs_2.indicies[idx2];
       //params.all->trace_message<<ec1pos<<" "<<ec2pos<<" "<<idx1<<" "<<idx2<<" "<<f->x<<" "<<ec2f->x<<endl;
       if(ec1pos < ec2pos) continue;
   
       while(ec1pos > ec2pos && ++idx2 < fs_2.size())
         ec2pos = fs_2.indicies[idx2];
   
       if(ec1pos == ec2pos)
       { //params.all->trace_message<<ec1pos<<" "<<ec2pos<<" "<<idx1<<" "<<idx2<<" "<<f->x<<" "<<ec2f->x<<endl;
         numint++;
         dotprod += fs_1.values[idx1] * fs_2.values[idx2];
         ++idx2;
       }
     }
     return dotprod;
   }

    float inner_prod(memory_tree& b, example* ec1, example* ec2)
    {
        flat_example* fec1 = flatten_sort_example(*b.all, ec1);
        flat_example* fec2 = flatten_sort_example(*b.all, ec2);
        float linear_prod = linear_kernel(fec1, fec2);
        linear_prod /= pow(fec1->total_sum_feat_sq*fec2->total_sum_feat_sq, 0.5);
        fec1->fs.delete_v(); 
        fec2->fs.delete_v();
        free(fec1);
        free(fec2);
        return linear_prod;
    }

    void init_tree(memory_tree& b)
    {
        //simple initilization: initilize the root only
        b.routers_used = 0;
        b.nodes.push_back(node());
        b.nodes[0].internal = -1; //mark the root as leaf
        b.nodes[0].base_router = (b.routers_used++);

        b.max_routers = b.max_nodes;
        cout<<"tree initiazliation is done...."<<endl
            <<"max nodes "<<b.max_nodes<<endl
            <<"tree size: "<<b.nodes.size()<<endl
            <<"learn at leaf: "<<b.learn_at_leaf<<endl
            <<"alpha: "<<b.alpha<<endl;
    }

    //rout based on the prediction
    inline uint32_t descent(node& n, const float prediction)
    { 
        //prediction <0 go left, otherwise go right
        if(prediction < 0){
            n.nl++; //increment the number of examples routed to the left.
            return n.left;
        }
        else{ //otherwise go right.
            n.nr++; //increment the number of examples routed to the right.
            return n.right; 
        }
    }

    //return the id of the example and the leaf id (stored in cn)
    inline int random_sample_example_pop(memory_tree& b, uint32_t& cn)
    {
        cn = 0; //always start from the root:
        while (b.nodes[cn].internal == 1)
        {
            float pred = 0.;   //deal with some edge cases:
            if (b.nodes[cn].nl < 1) //no examples routed to left ever:
                pred = 1.f; //go right.
            else if (b.nodes[cn].nr < 1) //no examples routed to right ever:
                pred = -1.f; //go left.
            else if ((b.nodes[cn].nl >= 1) && (b.nodes[cn].nr >= 1))
                pred = merand48(b.all->random_state) < (b.nodes[cn].nl*1./(b.nodes[cn].nr+b.nodes[cn].nl)) ? -1.f : 1.f;
            else{
                cout<<cn<<" "<<b.nodes[cn].nl<<" "<<b.nodes[cn].nr<<endl;
                cout<<"Error:  nl = 0, and nr = 0, exit...";
                exit(0);
            }
            
            if (pred < 0){
                b.nodes[cn].nl--;
                cn = b.nodes[cn].left; 
            }
            else{
                b.nodes[cn].nr--;
                cn = b.nodes[cn].right;
            }
        }

        if (b.nodes[cn].examples_index.size() >= 1){
            int loc_at_leaf = int(merand48(b.all->random_state)*b.nodes[cn].examples_index.size());
            pop_at_index(b.nodes[cn].examples_index, loc_at_leaf); 
            return loc_at_leaf;
        }
        else    
            return -1;
    }


    //train the node with id cn, using the statistics stored in the node to
    //formulate a binary classificaiton example.
    float train_node(memory_tree& b, base_learner& base, example& ec, const uint32_t cn)
    {
        //predict, learn and predict
        //note: here we first train the router and then predict.
        //MULTICLASS::label_t mc = ec.l.multi;
        //uint32_t save_multi_pred = ec.pred.scalar;
        MULTILABEL::labels multilabels = ec.l.multilabels;
        MULTILABEL::labels preds = ec.pred.multilabels;
        //ec.l.simple.label = 1.;
        ec.l.simple = {1.f, 1.f, 0.f};
        base.predict(ec, b.nodes[cn].base_router);
        float prediction = ec.pred.scalar; 
        float imp_weight = 1.f; //no importance weight.
        
        double weighted_value = (1.-b.alpha)*log(b.nodes[cn].nl/b.nodes[cn].nr*1.)/log(2.)+b.alpha*prediction;
        //b.nodes[cn].nl/b.nodes[cn].nr<<" "<<" "<<prediction<<" "<<weighted_value<<endl;
        float route_label = weighted_value < 0.f ? -1.f : 1.f;
        
        ec.l.simple = {route_label, imp_weight, 0.f};
        base.learn(ec, b.nodes[cn].base_router); //update the router according to the new example.
        
        base.predict(ec, b.nodes[cn].base_router);
        float save_binary_scalar = ec.partial_prediction;
        save_binary_scalar = ec.pred.scalar;
        //cout<<ec.partial_prediction<<" "<<ec.pred.scalar<<endl;
        //float save_binary_scalar = ec.pred.scalar;

        ec.pred.multilabels = preds;
        ec.l.multilabels = multilabels;
        //ec.l.multi = mc;
        //ec.pred.multiclass = save_multi_pred;
        return save_binary_scalar;
    }

    
    //turn a leaf into an internal node, and create two children
    //when the number of examples is too big
    void split_leaf(memory_tree& b, base_learner& base, const uint32_t cn)
    {
        //create two children
        b.nodes[cn].internal = 1; //swith to internal node.
        uint32_t left_child = (uint32_t)b.nodes.size();
        b.nodes.push_back(node());
        b.nodes[left_child].internal = -1;  //left leaf
        b.nodes[left_child].base_router = (b.routers_used++);
        uint32_t right_child = (uint32_t)b.nodes.size();
        b.nodes.push_back(node());  
        b.nodes[right_child].internal = -1;  //right leaf
        b.nodes[right_child].base_router = (b.routers_used++); 

        if (b.nodes[cn].depth + 1 > b.max_depth)
            b.max_depth = b.nodes[cn].depth + 1;
        //cout<<b.max_depth<<endl;

        b.nodes[cn].left = left_child;
        b.nodes[cn].right = right_child;
        b.nodes[left_child].parent = cn;
        b.nodes[right_child].parent = cn;
        b.nodes[left_child].depth = b.nodes[cn].depth + 1;
        b.nodes[right_child].depth = b.nodes[cn].depth + 1;

        if (b.nodes[left_child].depth > b.max_depth)
            b.max_depth = b.nodes[left_child].depth;

        //rout the examples stored in the node to the left and right
        for(size_t ec_id = 0; ec_id < b.nodes[cn].examples_index.size(); ec_id++) //scan all examples stored in the cn
        {
            uint32_t ec_pos = b.nodes[cn].examples_index[ec_id];
            MULTILABEL::labels multilabels = b.examples[ec_pos]->l.multilabels;
            MULTILABEL::labels preds = b.examples[ec_pos]->pred.multilabels;
            //MULTICLASS::label_t mc = b.examples[ec_pos]->l.multi;
            //uint32_t save_multi_pred = b.examples[ec_pos]->pred.multiclass;
            b.examples[ec_pos]->l.simple = {1.f, 1.f, 0.f};
            base.predict(*b.examples[ec_pos], b.nodes[cn].base_router); //re-predict
            float scalar = b.examples[ec_pos]->pred.scalar;
            if (scalar < 0)
            {
                b.nodes[left_child].examples_index.push_back(ec_pos);
                float leaf_pred = train_node(b, base, *b.examples[ec_pos], left_child);
                descent(b.nodes[left_child], leaf_pred); //fake descent, only for update nl and nr                
            }
            else
            {
                b.nodes[right_child].examples_index.push_back(ec_pos);
                float leaf_pred = train_node(b, base, *b.examples[ec_pos], right_child);
                descent(b.nodes[right_child], leaf_pred); //fake descent. for update nr and nl
            }
            b.examples[ec_pos]->pred.multilabels = preds;
            b.examples[ec_pos]->l.multilabels = multilabels;
            //b.examples[ec_pos]->l.multi = mc;
            //b.examples[ec_pos]->pred.multiclass = save_multi_pred;
        }
        b.nodes[cn].examples_index.delete_v(); //empty the cn's example list
        b.nodes[cn].nl = std::max(double(b.nodes[left_child].examples_index.size()), 0.001); //avoid to set nl to zero
        b.nodes[cn].nr = std::max(double(b.nodes[right_child].examples_index.size()), 0.001); //avoid to set nr to zero

        if (std::max(b.nodes[cn].nl, b.nodes[cn].nr) > b.max_ex_in_leaf)
        {
            b.max_ex_in_leaf = std::max(b.nodes[cn].nl, b.nodes[cn].nr);
            cout<<"max_ex_in_leaf: "<<b.max_ex_in_leaf<<" at node "<<cn<<endl;
        }

    }
    
    /*
    //pick up the "closest" example in the leaf using the score function.
    int64_t pick_nearest(memory_tree& b, base_learner& base, const uint32_t cn, example& ec)
    {
        if (b.nodes[cn].examples_index.size() > 0)
        {
            float max_score = -FLT_MAX;
            int64_t max_pos = -1;
            for(size_t i = 0; i < b.nodes[cn].examples_index.size(); i++)
            {
                float score = 0.f;
                uint32_t loc = b.nodes[cn].examples_index[i];

                if (b.learn_at_leaf == true){
                    //cout<<"learn at leaf"<<endl;
                    example* kprod_ec = &calloc_or_throw<example>();
                    diag_kronecker_product(ec, *b.examples[loc], *kprod_ec);
                    kprod_ec->l.simple = {-1., 1., 0.};
                    base.predict(*kprod_ec, b.max_routers);
                    //score = kprod_ec->pred.scalar;
                    score = kprod_ec->partial_prediction;
                    free_example(kprod_ec);
                }
                else
                    score = inner_prod(b, &ec, b.examples[loc]); 
                
                if (score > max_score){
                    max_score = score;
                    max_pos = (int64_t)loc;
                }
            }
            return max_pos;
        }
        else
            return -1;
    }*/

    void compute_scores_labels(memory_tree& b, base_learner& base, 
            const uint32_t cn, example& ec, int K, v_array<uint32_t>& top_K_labs)
    {   
        top_K_labs.delete_v();
        top_K_labs = v_init<uint32_t>();
        v_array<score_label> score_labs = v_init<score_label>();
        if (b.nodes[cn].examples_index.size() == 0){
            return;
        }
        
        for(size_t i = 0; i < b.nodes[cn].examples_index.size(); i++){
            float score = 0.f; //the similarity with respect to ec.
            uint32_t loc = b.nodes[cn].examples_index[i];
            if (b.learn_at_leaf == true){
                example* kprod_ec = &calloc_or_throw<example>();

                //MULTILABEL::labels k_multilabels = kprod_ec->l.multilabels;
                diag_kronecker_product(ec, *b.examples[loc], *kprod_ec);
                kprod_ec->l.simple = {1.f, 1., 0.};
                base.predict(*kprod_ec, b.max_routers);
                score = kprod_ec->partial_prediction;

                //kprod_ec->l.multilabels = k_multilabels;
                //kprod_ec->l.multilabels.label_v.delete_v();
                free_example(kprod_ec);
            } 
            else
                score = inner_prod(b, &ec, b.examples[loc]); 

            //insert labels:
            for (uint32_t lab : b.examples[loc]->l.multilabels.label_v){
                bool in = false;
                for (score_label& s_l : score_labs){
                    if (lab == s_l.label){ //this lab is in score_labs array
                        s_l.score += exp(5.*(score));
                        in = true;
                        break;
                    }
                }
                if (in == false)
                    score_labs.push_back(score_label(lab, score));   
            }
        }
        //get the top K ranked labels:
        if (score_labs.size() <= K){ //if less than K:
            for (auto l_s : score_labs)
                top_K_labs.push_back(l_s.label);
        }
        else{
            for (int pass = 0; pass < K; pass++){ //~O(Klog(N))
                uint32_t max_lab_loc = 0;
                float max_score = -FLT_MAX;
                for (size_t i = 0; i < score_labs.size(); i++){
                    score_label& l_s = score_labs[i];
                    if (l_s.score > max_score){
                        max_score = l_s.score;
                        max_lab_loc = i;
                    }
                }
                top_K_labs.push_back(score_labs[max_lab_loc].label);
                score_labs[max_lab_loc].score = -FLT_MAX; 
            }
        }
        score_labs.delete_v();
    }

    template<typename T> 
    inline bool in_v_array(const v_array<T>& array, const T& item, uint32_t& pos){
        pos = 0;
        for (uint32_t i = 0; i < array.size(); i++){
            if (array[i] == item){
                pos = i;
                return true;
            }
        }
        return false;
    }
    template<typename T>
    inline int over_lap(const v_array<T>& array_1, const v_array<T>& array_2){
        int num_overlap = 0;
        for (size_t i1 = 0; i1 < array_1.size(); i1++){
            T item1 = array_1[i1];
            uint32_t pos = 0;
            if (in_v_array(array_2, item1, pos) == true)
                num_overlap++;
        }
        return num_overlap;
    }

    void learn_similarity_at_leaf(memory_tree& b, base_learner& base, const uint32_t cn, example& ec)
    {
        //MULTILABEL::labels multilabels = ec.l.multilabels;
        //MULTILABEL::labels preds = ec.pred.multilabels;
        for (uint32_t loc : b.nodes[cn].examples_index)
        {
            example* ec_loc = b.examples[loc];
            example* kprod_ec = &calloc_or_throw<example>();

            diag_kronecker_product(ec, *ec_loc, *kprod_ec);

            //MULTILABEL::labels k_multilabels = kprod_ec->l.multilabels;
            //MULTILABEL::labels k_preds = kprod_ec->pred.multilabels;
        
            int num_overlap = over_lap(b.examples[loc]->l.multilabels.label_v, ec.l.multilabels.label_v);
            
            if (num_overlap > 0)
                //kprod_ec->l.simple = {float(1.*num_overlap), 1.f, 0.}; //weighted based on the number of overlap items.
                kprod_ec->l.simple = {1.f, float(1.*num_overlap), 0.}; //weighted based on the number of overlap items.
            else
                kprod_ec->l.simple = {-1.f, 1., 0.}; //reward = 0:
                //kprod_ec->l.simple = {0.f, 1.f, 0.};
    
            base.learn(*kprod_ec, b.max_routers);

            //ec.pred.multilabels = preds;
            //ec.l.multilabels = multilabels;

            //kprod_ec->pred.multilabels = k_preds;
            //kprod_ec->l.multilabels = k_multilabels; 
            //kprod_ec->l.multilabels.label_v.delete_v();
            free_example(kprod_ec);
        }
    } 

    inline float compute_precision_at_K(example& ec){
        float pk_score = 0;
        uint32_t tmp_pos = 0;
        for (uint32_t label : ec.pred.multilabels.label_v){
            if (in_v_array(ec.l.multilabels.label_v, label, tmp_pos))
                pk_score++;
        }
        if (ec.pred.multilabels.label_v.size() == 0)
            pk_score = 0;
        else
            pk_score /= (ec.pred.multilabels.label_v.size()*1.);
        
        return pk_score;
    }

    void predict(memory_tree& b, base_learner& base, example& ec)
    {
        //example& ec = calloc_or_throw<example>();
        //copy_example_data(&ec, &test_ec);
        //remove_repeat_features_in_ec(ec);
        
        MULTILABEL::labels multilabels = ec.l.multilabels;
        MULTILABEL::labels preds = ec.pred.multilabels;
        uint32_t cn = 0;
        ec.l.simple = {1., 1.f, 0.f};
        while(b.nodes[cn].internal == 1) //if it's internal
        {
            base.predict(ec, b.nodes[cn].base_router);
            uint32_t newcn = ec.pred.scalar < 0 ? b.nodes[cn].left : b.nodes[cn].right; //do not need to increment nl and nr.
            cn = newcn;
        }
        ec.pred.multilabels = preds; 
        ec.l.multilabels = multilabels;

        v_array<uint32_t> top_K_labels = v_init<uint32_t>();
        compute_scores_labels(b, base, cn, ec, b.K, top_K_labels);
    
        ec.pred.multilabels = preds; 
        ec.l.multilabels = multilabels;
        
        //compute precision @ K:
        copy_array(ec.pred.multilabels.label_v, top_K_labels);
        float p_at_k = compute_precision_at_K(ec);
        b.num_mistakes += (1 - p_at_k);
        top_K_labels.delete_v();
        //free_example(&ec);
    }

    //node here the ec is already stored in the b.examples, the task here is to rout it to the leaf, 
    //and insert the ec_array_index to the leaf.
    void insert_example(memory_tree& b, base_learner& base, const uint32_t& ec_array_index, bool fake_insert = false)
    {
        uint32_t cn = 0; //start from the root.
        while(b.nodes[cn].internal == 1) //if it's internal node:
        {   
            //predict and train the node at cn.
            float router_pred = train_node(b, base, *b.examples[ec_array_index], cn); 
            uint32_t newcn = descent(b.nodes[cn], router_pred); //updated nr or nl
            cn = newcn; 
        }
        if ((b.nodes[cn].internal == -1) && (b.learn_at_leaf == true)){ //if reach leaf, we learn similarity:
            learn_similarity_at_leaf(b, base, cn, *b.examples[ec_array_index]);  //learn similarity function at leafb.l
        }

        //insert the example in leaf and deal with split:
        if((b.nodes[cn].internal == -1) && (fake_insert == false)) //get to leaf:
        {   
            b.nodes[cn].examples_index.push_back(ec_array_index);
            float leaf_pred = train_node(b, base, *b.examples[ec_array_index], cn); //tain the leaf as well.
            descent(b.nodes[cn], leaf_pred); //this is a faked descent, the purpose is only to update nl and nr of cn

            //if the number of examples exceeds the max_leaf_examples, and not reach the max_nodes - 2 yet, we split:
            if((b.nodes[cn].examples_index.size() >= b.max_leaf_examples) && (b.nodes.size() + 2 <= b.max_nodes))
                split_leaf(b, base, cn); 
        }
    }

    void experience_replay(memory_tree& b, base_learner& base)
    {
        uint32_t cn = 0; //start from root, randomly descent down! 
        int loc_at_leaf = random_sample_example_pop(b, cn);
        if (loc_at_leaf >= 0){
            uint32_t ec_id = b.nodes[cn].examples_index[loc_at_leaf]; //ec_id is the postion of the sampled example in b.examples. 
            //re-insert:note that we do not have to 
            //restore the example into b.examples, as it's alreay there
            insert_example(b, base, ec_id); 
        }

    }

    //learn: descent the example from the root while generating binary training
    //example for each node, including the leaf, and store the example at the leaf.
    void learn(memory_tree& b, base_learner& base, example& ec)
    {        
        if (b.test_mode == false){
            b.iter++;

            predict(b, base, ec);
            if (b.iter%5000 == 0)
                cout<<"at iter "<<b.iter<<", pred error: "<<b.num_mistakes*1./b.iter<<endl;
            
            example* new_ec = &calloc_or_throw<example>();
            copy_example_data(new_ec, &ec);
            //remove_repeat_features_in_ec(*new_ec); ////sort unique.
            
            b.examples.push_back(new_ec);   
            b.num_ecs++; 

            float random_prob = merand48(b.all->random_state);
            if (true){
                insert_example(b, base, b.examples.size()-1);
            }
            else{
                insert_example(b, base, b.examples.size()-1, true);
                b.num_ecs--;
                free_example(new_ec);
                b.examples.pop();
            }

            //if (b.iter % 1 == 0)
            for (int i = 0; i < 1; i++)
                experience_replay(b, base);   
        }
        else if (b.test_mode == true){
            b.iter++;
            if (b.iter % 5000 == 0)
                cout<<"at iter "<<b.iter<<", pred error: "<<b.num_mistakes*1./b.iter<<endl;
            predict(b, base, ec);
        }

    } 

    /////////////////////////////output stuff///////////////
    bool is_test_label(MULTILABEL::labels& ld)
    { if (ld.label_v.size() == 0)
        return true;
    else
        return false;
    }

    void print_update(vw& all, bool is_test, example& ec)
    { if (all.sd->weighted_examples() >= all.sd->dump_interval && !all.quiet && !all.bfgs)
    { stringstream label_string;
        if (is_test)
        label_string << " unknown";
        else
        for(size_t i = 0; i < ec.l.multilabels.label_v.size(); i++)
            label_string << " " << ec.l.multilabels.label_v[i];

        stringstream pred_string;
        for(size_t i = 0; i < ec.pred.multilabels.label_v.size(); i++)
        pred_string << " " << ec.pred.multilabels.label_v[i];

        all.sd->print_update(all.holdout_set_off, all.current_pass, label_string.str(), pred_string.str(),
                         ec.num_features, all.progress_add, all.progress_arg);
    }
    }



    void output_example(vw& all, example& ec)
    { 
        MULTILABEL::labels& ld = ec.l.multilabels;
        float loss = 0.;
        if (!is_test_label(ld))
        { //need to compute exact loss
            float pk_score = compute_precision_at_K(ec);
            loss =+ (1. - pk_score) * ec.weight;
        }

        all.sd->update(ec.test_only, !is_test_label(ld), loss, 1.f, ec.num_features);

        for (int sink : all.final_prediction_sink)
            if (sink >= 0)
        { std::stringstream ss;

        for (size_t i = 0; i < ec.pred.multilabels.label_v.size(); i++)
        { if (i > 0)
              ss << ',';
            ss << ec.pred.multilabels.label_v[i];
        }
        ss << ' ';
        all.print_text(sink, ss.str(), ec.tag);
        }

    print_update(all, is_test_label(ec.l.multilabels), ec);
    }

    void finish_example(vw& all, memory_tree&, example& ec)
    {
        output_example(all, ec);
        VW::finish_example(all, &ec);
    }

    void finish(memory_tree& b)
    {
        for (size_t i = 0; i < b.nodes.size(); ++i)
            b.nodes[i].examples_index.delete_v();
        b.nodes.delete_v();
        for (size_t i = 0; i < b.examples.size(); i++)
            free_example(b.examples[i]);
        b.examples.delete_v();
        cout<<b.max_nodes<<endl;
    }

    ///////////////////Save & Load//////////////////////////////////////
    ////////////////////////////////////////////////////////////////////
    #define writeit(what,str)                               \
    do                                                    \
        {                                                   \
        msg << str << " = " << what << " ";               \
        bin_text_read_write_fixed (model_file,            \
                                    (char*) &what,         \
                                    sizeof (what),         \
                                    "",                    \
                                    read,                  \
                                    msg,                   \
                                    text);                 \
        }                                                   \
    while (0);

    #define writeitvar(what,str,mywhat)                     \
    auto mywhat = (what);                                 \
    do                                                    \
        {                                                   \
        msg << str << " = " << mywhat << " ";             \
        bin_text_read_write_fixed (model_file,            \
                                    (char*) &mywhat,       \
                                    sizeof (mywhat),       \
                                    "",                    \
                                    read,                  \
                                    msg,                   \
                                    text);                 \
        }                                                   \
    while (0);

    void save_load_example(example* ec, io_buf& model_file, bool& read, bool& text, stringstream& msg)
    {   //deal with tag
        //deal with labels:
        //writeit(ec->l.multi.label, "multiclass_label");
        //writeit(ec->l.multi.weight, "multiclass_weight");
        writeit(ec->num_features, "num_features");
        writeit(ec->total_sum_feat_sq, "total_sum_features");
        writeit(ec->weight, "example_weight");
        writeit(ec->loss, "loss");
        writeit(ec->ft_offset, "ft_offset");
        writeitvar(ec->l.multilabels.label_v.size(), "label_size", label_size);
        if (read){
            ec->l.multilabels.label_v.erase();
            for (uint32_t i = 0; i < label_size; i++)
                ec->l.multilabels.label_v.push_back(0);
        }
        for (uint32_t i = 0; i < label_size; i++)
            writeit(ec->l.multilabels.label_v[i], "ec_label");

        writeitvar(ec->tag.size(), "tags", tag_number);
        if (read){
            ec->tag.erase();
            for (uint32_t i = 0; i < tag_number; i++)
                ec->tag.push_back('a');
        }
        for (uint32_t i = 0; i < tag_number; i++)
            writeit(ec->tag[i], "tag");
        
        //deal with tag:
        writeitvar(ec->indices.size(), "namespaces", namespace_size);
        if (read){
            ec->indices.delete_v();
            for (uint32_t i = 0; i< namespace_size; i++){
                ec->indices.push_back('\0');
            }
        }
        for(uint32_t i = 0; i < namespace_size; i++)
            writeit(ec->indices[i], "namespace_index");

        //deal with features
        for (namespace_index nc: ec->indices){
            features* fs = &ec->feature_space[nc];
            writeitvar(fs->size(), "features_", feat_size);
            if (read){
                fs->erase();
                fs->values = v_init<feature_value>();
                fs->indicies = v_init<feature_index>();
                for (uint32_t f_i = 0; f_i < feat_size; f_i++){
                    fs->push_back(0, 0);
                }
            }
            for (uint32_t f_i = 0; f_i < feat_size; f_i++)
                writeit(fs->values[f_i], "value");
            for (uint32_t f_i = 0; f_i < feat_size; f_i++)
                writeit(fs->indicies[f_i], "index");
        }
    }

    void save_load_node(node& cn, io_buf& model_file, bool& read, bool& text, stringstream& msg)
    {
        writeit(cn.parent, "parent");
        writeit(cn.internal, "internal");
        writeit(cn.depth, "depth");
        writeit(cn.base_router, "base_router");
        writeit(cn.left, "left");
        writeit(cn.right, "right");
        writeit(cn.nl, "nl");
        writeit(cn.nr, "nr");
        writeitvar(cn.examples_index.size(), "leaf_n_examples", leaf_n_examples);
        if (read){
            cn.examples_index.erase();
            for (uint32_t k = 0; k < leaf_n_examples; k++)
                    cn.examples_index.push_back(0);
        }
        for (uint32_t k = 0; k < leaf_n_examples; k++)
            writeit(cn.examples_index[k], "example_location");
    }

    void save_load_memory_tree(memory_tree& b, io_buf& model_file, bool read, bool text)
    {
        stringstream msg;
        if (model_file.files.size() > 0){
            if (read)
                b.test_mode = true;

            if (read)
            {
                size_t ss = 0;
                writeit(ss, "stride_shift");
                b.all->weights.stride_shift(ss);
            }
            else
            {
                size_t ss = b.all->weights.stride_shift();
                writeit(ss, "stride_shift");
            }
            
            writeit(b.max_nodes, "max_nodes");
            writeit(b.learn_at_leaf, "learn_at_leaf");
            writeitvar(b.nodes.size(), "nodes", n_nodes); 

            if (read){
                b.nodes.erase();
                for (uint32_t i = 0; i < n_nodes; i++)
                    b.nodes.push_back(node());
            }
            
            //node  
            for(uint32_t i = 0; i < n_nodes; i++){
                save_load_node(b.nodes[i], model_file, read, text, msg);
            }
            //deal with examples:
            writeitvar(b.examples.size(), "examples", n_examples);
            if (read){
                b.examples.erase();
                for (uint32_t i = 0; i < n_examples; i++){
                    example* new_ec = &calloc_or_throw<example>();
                    b.examples.push_back(new_ec);
                }
            }
            for (uint32_t i = 0; i < n_examples; i++)
                save_load_example(b.examples[i], model_file, read, text, msg);
            
            
        }
    }
    //////////////////////////////End of Save & Load///////////////////////////////
} //namespace

base_learner* memory_tree_xml_setup(vw& all)
{
    using namespace memory_tree_xml_ns;
    if (missing_option<uint32_t, true>(all, "memory_tree_xml", "Make a memory tree with at most <n> nodes"))
        return nullptr;
    
    new_options(all, "memory tree xml options")
      ("leaf_example_multiplier", po::value<uint32_t>()->default_value(1.0), "multiplier on examples per leaf (default = log nodes)")
      ("learn_at_leaf", po::value<bool>()->default_value(true), "whether or not learn at leaf (defualt = True)")
      ("Precision_at_K", po::value<int>()->default_value(1), "Precision@K (defualt K = 1)")
      ("Alpha", po::value<float>()->default_value(0.1), "Alpha");
     add_options(all);

    po::variables_map& vm = all.vm;

    memory_tree& tree = calloc_or_throw<memory_tree>();
    tree.all = &all;
    tree.max_nodes = vm["memory_tree_xml"].as<uint32_t>();
    tree.learn_at_leaf = vm["learn_at_leaf"].as<bool>();
    tree.K = vm["Precision_at_K"].as<int>();

    if (vm.count("leaf_example_multiplier"))
      {
	tree.max_leaf_examples = vm["leaf_example_multiplier"].as<uint32_t>() * (log(tree.max_nodes)/log(2));
	*all.file_options << " --leaf_example_multiplier " << vm["leaf_example_multiplier"].as<uint32_t>();
      }
    if (vm.count("Alpha"))
      {
	tree.alpha = vm["Alpha"].as<float>();
	*all.file_options << " --Alpha " << tree.alpha;
      }
    
    init_tree(tree);

    if (! all.quiet)
        all.trace_message << "memory_tree_xml:" << " "
                    <<"max_nodes = "<< tree.max_nodes << " " 
                    <<"max_leaf_examples = "<<tree.max_leaf_examples<<" "
                    <<"alpha = "<<tree.alpha<<" "
                    <<"Precision@K, K = "<<tree.K
                    <<std::endl;
    
    learner<memory_tree>& l = 
        init_learner(&tree, setup_base(all), learn, predict, tree.max_routers,
                    prediction_type::multilabels);
    
    l.set_save_load(save_load_memory_tree);
    l.set_finish_example(finish_example);
    l.set_finish(finish);

    all.p->lp = MULTILABEL::multilabel;
    all.label_type = label_type::multi;
    all.delete_prediction = MULTILABEL::multilabel.delete_label;


    //srand(time(0));
    return make_base (l);
}







//learning rate for aloi: 0.0001