#include <algorithm>
#include <cmath>
#include <cstdio>
#include <float.h>
#include <time.h>
#include <sstream>
#include <ctime>
#include "reductions.h"
#include "rand48.h"
#include "vw.h"
#include "bleu.h"


using namespace std;
using namespace LEARNER;

namespace nearest_neighbor_ns
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

    void copy_example_data(example* dst, example* src, bool no_feat = false)
    { 
        dst->l = src->l;
        dst->l.multi.label = src->l.multi.label;

        copy_array(dst->tag, src->tag);
        dst->example_counter = src->example_counter;
        dst->ft_offset = src->ft_offset;

        if (no_feat == false){
            copy_array(dst->indices, src->indices);
            for (namespace_index c : src->indices)
                dst->feature_space[c].deep_copy_from(src->feature_space[c]);
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


    float inner_product_two_features(const features& fs_1, const features& fs_2, double norm_sq1 = 1., double norm_sq2 = 1.)
    {
        float dotprod = 0.0;
        if (fs_2.indicies.size() == 0)
            return 0.f;
        int numint = 0;
        for (size_t idx1 = 0, idx2 = 0; idx1 < fs_1.size() && idx2 < fs_2.size() ; idx1++)
        { 
            uint64_t ec1pos = fs_1.indicies[idx1];
            uint64_t ec2pos = fs_2.indicies[idx2];
            
            if(ec1pos < ec2pos) continue;
   
            while(ec1pos > ec2pos && ++idx2 < fs_2.size())
                ec2pos = fs_2.indicies[idx2];
   
            if(ec1pos == ec2pos)
            { 
                numint++;
                dotprod += fs_1.values[idx1] * fs_2.values[idx2];
                ++idx2;
            }
        }
        return (dotprod/pow(norm_sq1*norm_sq2, 0.5));
    }

    float linear_kernel(const flat_example* fec1, const flat_example* fec2)
    { 
        float dotprod = 0;
        features& fs_1 = (features&)fec1->fs;
        features& fs_2 = (features&)fec2->fs;
        dotprod = inner_product_two_features(fs_1,fs_2);
        return dotprod;
   }

    inline void free_example(example* ec)
    {
        ec->tag.delete_v();
        if (ec->passthrough){
            ec->passthrough->delete_v();
        }
        for (int i = 0; i < 256; i++)
            ec->feature_space[i].delete_v();
        ec->indices.delete_v();
        free(ec);
    }



    void remove_repeat_features_in_f(features& f, float& total_sum_feat_sq )
    {
        for (size_t i = 0; i < f.indicies.size(); i++){
            if (f.values[i] != -FLT_MAX){
                uint64_t loc = f.indicies[i];
                for (size_t j = i+1; j < f.indicies.size(); j++){
                    if (loc == f.indicies[j]){
                        total_sum_feat_sq -= (f.values[i]*f.values[i] + f.values[j]*f.values[j]);
                        f.values[i] += f.values[j];
                        total_sum_feat_sq += f.values[i]*f.values[i];
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
            remove_repeat_features_in_f(ec.feature_space[nc], ec.total_sum_feat_sq);
    }




    inline float square_norm_feature(features& fs){
        double total_feat_sum_sq = 0.;
        for (size_t t = 0; t < fs.values.size(); t++)
            total_feat_sum_sq += pow(fs.values[t],2);
        return total_feat_sum_sq;
    }

    float compute_similarity_under_namespace(example& ec1, example& ec2, unsigned char ns)
    {
        features& f1 = ec1.feature_space[ns];
        features& f2 = ec2.feature_space[ns];
        double f1_feat_sum_sq = square_norm_feature(f1);
        double f2_feat_sum_sq = square_norm_feature(f2);
        
        //cout<<f1.values.size()<<" "<<f2.values.size()<<endl;
        //compute the innner product between these twos:
        float dotprod = inner_product_two_features(f1, f2, 1., 1.);
        //float dotprod = inner_product_two_features(f1, f2, f1_feat_sum_sq, f2_feat_sum_sq);
        //size_t d = f1.values.size();
        //float dotprod = 0.;
        //for (size_t i = 0; i < d; i++)
        //    dotprod += f1.values[i]*f2.values[i];
        
        return dotprod;
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

    //memory_tree
    struct memory_tree
    {
        vw* all;

        v_array<node> nodes;  //array of nodes.
        v_array<example*> examples; //array of example points
        
        size_t max_leaf_examples; 
        uint32_t task_id;
        size_t max_nodes;
        size_t max_routers;
        float alpha; //for cpt type of update.
        size_t routers_used;
        int iter;

        size_t max_depth;
        size_t max_ex_in_leaf;


        bool path_id_feat;

        uint32_t num_mistakes;
        uint32_t num_ecs;
        uint32_t num_test_ecs;
        uint32_t test_mistakes;
        bool learn_at_leaf;

        float total_reward;
        bool test_mode;
        float total_test_reward;
        unsigned char Q;
        unsigned char A;
        float construct_time;
        float test_time;

        memory_tree()
        {
            construct_time = 0.;
            test_time = 0.;
            nodes = v_init<node>();
            examples = v_init<example*>();
            alpha = 0.5;
            routers_used = 0;
            iter = 0;
            num_mistakes = 0;
            num_ecs = 0;
            num_test_ecs = 0;
            path_id_feat = false;
            test_mode = false;
            max_depth = 0;
            max_ex_in_leaf = 0;
            total_reward = 0;
            total_test_reward = 0.;
        }
    };


    float normalized_linear_prod(memory_tree& b, example* ec1, example* ec2)
    {
        if (b.task_id == 1){
            flat_example* fec1 = flatten_sort_example(*b.all, ec1);
            flat_example* fec2 = flatten_sort_example(*b.all, ec2);
            float linear_prod = linear_kernel(fec1, fec2);
            fec1->fs.delete_v(); 
            fec2->fs.delete_v();
            free(fec1);
            free(fec2);
            return linear_prod/pow(fec1->total_sum_feat_sq*fec2->total_sum_feat_sq, 0.5);
        }
        else if (b.task_id != 1){
            double f1_feat_sum_sq = square_norm_feature(ec1->feature_space[b.Q]);
            double f2_feat_sum_sq = square_norm_feature(ec2->feature_space[b.Q]);
            float linear_prod = inner_product_two_features(ec1->feature_space[b.Q], ec2->feature_space[b.Q], f1_feat_sum_sq,f2_feat_sum_sq); //joints.
            return linear_prod;
        }
        return 0;
    }


    void init_tree(memory_tree& b)
    {
        srand48(4000);
        //simple initilization: initilize the root only
        b.routers_used = 0;
        b.nodes.push_back(node());
        b.nodes[0].internal = -1; //mark the root as leaf
        b.nodes[0].base_router = (b.routers_used++);

        b.max_routers = b.max_nodes;
        b.Q = 'J';
        b.A = 'I';

        cout<<"tree initiazliation is done...."<<endl
            <<"max nodes "<<b.max_nodes<<endl
            <<"tree size: "<<b.nodes.size()<<endl
            <<"learn at leaf: "<<b.learn_at_leaf<<endl
            <<"Taks id: "<<b.task_id<<endl
            <<"b.Q: "<<b.Q<<endl
            <<"b.A: "<<b.A<<endl;
    }




    //pick up the "closest" example in the leaf using the score function.
    int64_t pick_nearest(memory_tree& b, base_learner& base, example& ec)
    {
        if (b.examples.size() > 0)
        {
            float max_score = -FLT_MAX;
            int64_t max_pos = -1;
            for(size_t i = 0; i < b.examples.size(); i++)
            {
                //cout<<i<<endl;
                float score = 0.f;
                uint32_t loc = i;

                score = normalized_linear_prod(b, &ec, b.examples[loc]); //task 1: innner product between features. task 2: inner product between Qs.
                
                if (score > max_score){
                    max_score = score;
                    max_pos = (int64_t)loc;
                }
            }
            return max_pos;
        }
        else
            return -1;
    }

    float get_reward(memory_tree& b, example& ec, example& retrieved_ec){
        float reward = 0.;
        if (b.task_id == 1)
        {    //multi-class:
            if (ec.l.multi.label == retrieved_ec.l.multi.label)
                reward = 1.;
            else
                reward = 0.;
        }
        else if (b.task_id == 2){ //J and I
            reward = compute_similarity_under_namespace(ec, retrieved_ec, b.A);  //use the image part to compute the reward.
        }
        else if (b.task_id == 3){
            reward = bleu(ec.feature_space[b.A].indicies, retrieved_ec.feature_space[b.A].indicies,2);
        }
        return reward;
    }



    void predict(memory_tree& b, base_learner& base, example& test_ec)
    {
        example& ec = calloc_or_throw<example>();
        if (b.task_id == 1)
            copy_example_data(&ec, &test_ec);
        else if (b.task_id != 1){
            copy_example_data(&ec, &test_ec, true);
            ec.indices.push_back(b.Q);
            ec.feature_space[b.Q].deep_copy_from(test_ec.feature_space[b.Q]); //use joints.
        }
        
        int64_t closest_ec = pick_nearest(b, base, ec);
        //cout<<closest_ec<<endl;
        if (closest_ec != -1){
            float reward = get_reward(b, test_ec, *b.examples[closest_ec]);
            ec.pred.multiclass = b.examples[closest_ec]->l.multi.label;
            test_ec.pred.multiclass = b.examples[closest_ec]->l.multi.label;
            test_ec.loss = -reward * test_ec.weight;
            b.total_reward += reward;

            //if (b.iter > 7000*2)
            //if (b.iter > 82784*1)
            if (b.iter > 173551)
                b.total_test_reward += reward;
        }
        else{
            b.total_reward += 0.0;
            test_ec.loss = -1.*test_ec.weight;
        }

        free_example(&ec);
    }


    //learn: descent the example from the root while generating binary training
    //example for each node, including the leaf, and store the example at the leaf.
    void learn(memory_tree& b, base_learner& base, example& ec)
    {   
        //int32_t train_N = 7000*2;
        //int32_t train_N = 82784*1;
        int32_t train_N = 173551;
        b.iter++;       
        //if (b.test_mode == false){
        if (b.iter<=train_N){
            //predict(b, base, ec);
            if (b.iter%5000 == 0)
                //cout<<"at iter "<<b.iter<<", pred error: "<<b.num_mistakes*1./b.iter<<endl;
                cout<<"at iter "<<b.iter<<", average reward: "<<b.total_reward*1./b.iter<<endl;
            
            clock_t begin = clock();
            example* new_ec = &calloc_or_throw<example>();
            copy_example_data(new_ec, &ec);
            b.examples.push_back(new_ec);   
            b.num_ecs++; 
            clock_t end = clock();
            double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
            b.construct_time += elapsed_secs;
        }
        //else if (b.test_mode == true){
        else if (b.iter > train_N){
            if ((b.iter-train_N) % 10 == 0)
                cout<<"at iter "<<b.iter-train_N<<",total_reward "<<b.total_test_reward<<", avg_reward: "<<b.total_test_reward*1./(b.iter-train_N)<<endl;

            //for (size_t i = 0; i < ec.feature_space[b.A].values.size(); i++)
            //    cout<<ec.feature_space[b.A].indicies[i]<<" "<<ec.feature_space[b.A].values[i]<<endl;
            //exit(1);
            clock_t begin = clock();
            predict(b, base, ec);
            clock_t end = clock();
            double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
            b.test_time += elapsed_secs;
        }

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
        cout<<b.total_test_reward<<endl;
        cout<<"construct time: "<<b.construct_time<<endl;
        cout<<"test time: "<<b.test_time<<endl;
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
        writeit(ec->l.multi.label, "multiclass_label");
        writeit(ec->l.multi.weight, "multiclass_weight");
        writeit(ec->num_features, "num_features");
        writeit(ec->total_sum_feat_sq, "total_sum_features");
        writeit(ec->weight, "example_weight");
        writeit(ec->loss, "loss");
        writeit(ec->ft_offset, "ft_offset");

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
            writeit(b.Q, "Q");
            writeit(b.A, "A");
            writeit(b.task_id, "Task id");
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

base_learner* nearest_neighbor_setup(vw& all)
{
    using namespace nearest_neighbor_ns;
    if (missing_option<uint32_t, true>(all, "nearest_neighbor", "Make a nearest neighbor with at most <n> nodes"))
        return nullptr;
    
    new_options(all, "nearest neighbor options")
      ("leaf_example_multiplier", po::value<uint32_t>()->default_value(1.0), "multiplier on examples per leaf (default = log nodes)")
      ("learn_at_leaf", po::value<bool>()->default_value(true), "whether or not learn at leaf (defualt = True)")
      ("task", po::value<uint32_t>()->default_value(1.), "task: 1:multiclass; 2: Robot Configuration; 3:Q&A with bleu as reward")
      ("Alpha", po::value<float>()->default_value(0.1), "Alpha");
     add_options(all);

    po::variables_map& vm = all.vm;
    memory_tree& tree = calloc_or_throw<memory_tree> ();
    tree.all = &all;
    tree.max_nodes = vm["nearest_neighbor"].as<uint32_t>();
    tree.learn_at_leaf = vm["learn_at_leaf"].as<bool>();

    if (vm.count("leaf_example_multiplier"))
      {
	tree.max_leaf_examples = vm["leaf_example_multiplier"].as<uint32_t>() * (log(tree.max_nodes)/log(2));
	*all.file_options << " --leaf_example_multiplier " << vm["leaf_example_multiplier"].as<uint32_t>();
      }
    
    if (vm.count("task"))
    {
        tree.task_id = vm["task"].as<uint32_t>();
        *all.file_options << " --task "<< vm["task"].as<uint32_t>();
    }

    if (vm.count("Alpha"))
      {
	tree.alpha = vm["Alpha"].as<float>();
	*all.file_options << " --Alpha " << tree.alpha;
      }
    
    init_tree(tree);

    if (! all.quiet)
        all.trace_message << "nearest_neighbor:" << " "
                    <<"max_nodes = "<< tree.max_nodes << " " 
                    <<"max_leaf_examples = "<<tree.max_leaf_examples<<" "
                    <<"alpha = "<<tree.alpha<<" "
                    <<"Task = "<<tree.task_id
                    <<std::endl;
    
    learner<memory_tree>& l = 
        init_multiclass_learner (&tree, 
                setup_base (all),
                learn,
                predict,
                all.p, 
                tree.max_nodes + 1);
    
    //srand(time(0));
    l.set_save_load(save_load_memory_tree);
    l.set_finish(finish);

    return make_base (l);
}







//learning rate for aloi: 0.0001
/*
//f = f1 - f2
    void subtract_features(features& f1, features& f2, features& f)
    {//f1 and f2 are features under the same namespace
        f.deep_copy_from(f1);
        //now scan through f2's indices, check if the index is in f1's indices:
        for (size_t i = 0; i < f2.indicies.size(); i++){
            uint64_t ind_f2 = f2.indicies[i];
            float val_f2 = f2.values[i];
            uint64_t pos = 0;
            for (pos = 0; pos < f.indicies.size(); pos++)
            {
                if (ind_f2 == f.indicies[pos]) //the feature index of f2 is also in f1, then we subtract:
                {
                    f.values[pos] -= val_f2;
                    break;
                }
            }
            if (pos == f.indicies.size()) //this feature index of f2 is not in f1,  then push back (0-val), as we are doing f1-f2.
                f.push_back(0. - val_f2, ind_f2); 
        }
    }


    //ec1 - ec2
    void subtract_two_examples(example& ec1, example& ec2, example* subtracted_ec)
    {
        //copy tag from ec1 and ec2:
        copy_array(subtracted_ec->tag, ec1.tag);
        for (auto tag: ec2.tag)  //char
        {
	  size_t i = 0;
            for(i = 0; i < subtracted_ec->tag.size(); i++){
                if (tag == subtracted_ec->tag[i]) 
                    break;
            }
            if (i == subtracted_ec->tag.size())
                subtracted_ec->tag.push_back(tag);
        }
        //copy indices (different indices, no repeat)
        copy_array(subtracted_ec->indices, ec1.indices);
        for (auto c : ec2.indices){ //char: namespace_index
            size_t i = 0;
            for (i = 0; i < subtracted_ec->indices.size(); i++){
                if (c == subtracted_ec->indices[i])
                    break;
            }
            if (i == subtracted_ec->indices.size())
                subtracted_ec->indices.push_back(c);
        }
        //copy features (f1 - f2) 
        subtracted_ec->ft_offset = ec1.ft_offset; //0
        subtracted_ec->num_features = 0;
        subtracted_ec->total_sum_feat_sq = 0; 
        for (auto c : subtracted_ec->indices) //namespace index:
        {
            //cout<<"at "<<c<<endl;
            size_t pos1 = 0;
            for (pos1 = 0; pos1 < ec1.indices.size(); pos1++){
                if (c == ec1.indices[pos1])
                    break;
            }
            size_t pos2 = 0;
            for (pos2 = 0; pos2 < ec2.indices.size(); pos2++){
                if (c == ec2.indices[pos2])
                    break;
            }

            if ((pos1 < ec1.indices.size()) && (pos2 < ec2.indices.size())) //common name space:
                subtract_features(ec1.feature_space[c], ec2.feature_space[c], subtracted_ec->feature_space[c]);
            else if ((pos1 < ec1.indices.size()) && (pos2 == ec2.indices.size())) //f1 has this a name space that doesn't exist in f2:
                subtracted_ec->feature_space[c].deep_copy_from(ec1.feature_space[c]);
            else if ((pos1 == ec1.indices.size()) && (pos2 < ec2.indices.size())){
                subtracted_ec->feature_space[c].deep_copy_from(ec2.feature_space[c]);
                //negate the values
                for (size_t t = 0; t < subtracted_ec->feature_space[c].values.size(); t++)
                    subtracted_ec->feature_space[c].values[t] *= -1.;
            }
            subtracted_ec->num_features += subtracted_ec->feature_space[c].indicies.size();

            //update total_feature_square
            for (size_t v = 0; v < subtracted_ec->feature_space[c].values.size(); v++){
                subtracted_ec->total_sum_feat_sq += pow(subtracted_ec->feature_space[c].values[v],2);
                //cout<<pow(subtracted_ec->feature_space[c].values[v],2)<<endl;
            }
        }

        //otherstuff:
        subtracted_ec->partial_prediction = 0.0;
        subtracted_ec->passthrough = nullptr;
        subtracted_ec->loss = 0.;
        subtracted_ec->weight = ec1.weight;
        subtracted_ec->confidence = ec1.confidence;
        subtracted_ec->test_only = ec1.test_only;
        subtracted_ec->end_pass = ec1.end_pass;
        subtracted_ec->sorted = false;
        subtracted_ec->in_use = false;
    }
*/
