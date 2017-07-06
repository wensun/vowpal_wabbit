#include <algorithm>
#include <cmath>
#include <cstdio>
#include <float.h>
#include <time.h>
#include <sstream>
#include <random>

#include "reductions.h"
#include "rand48.h"


using namespace std;
using namespace LEARNER;
//using namespace VW;

namespace memory_tree_ns
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

    //f = f1 - f2
    void subtract_features(features& f1, features& f2, features& f)
    {//f1 and f2 are features under the same namespace
        f.deep_copy_from(f1);
        //now scan through f2's indices, check if the index is in f1's indices:
        for (int i = 0; i < f2.indicies.size(); i++){
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
            int i = 0;
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
            int i = 0;
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
            int pos1 = 0;
            for (pos1 = 0; pos1 < ec1.indices.size(); pos1++){
                if (c == ec1.indices[pos1])
                    break;
            }
            int pos2 = 0;
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


    ////Implement kronecker_product between two examples:
    //kronecker_prod at feature level:
    void diag_kronecker_prod_fs(features& f1, features& f2, float& total_sum_feat_sq)
    {
        for (size_t i1 = 0; i1 < f1.indicies.size(); i1++){
            size_t i2 = 0;
            for (i2 = 0; i2< f2.indicies.size(); i2++){
                if (f1.indicies[i1] == f2.indicies[i2]){
                    f1.values[i1] *= f2.values[i2];
                    total_sum_feat_sq += pow(f1.values[i1],2);
                    break;
                }
            }
            if (i2 == f2.indicies.size()){ //f1's index does not appear in f2, namely value of the index in f2 is zero.
                f1.values[i1] = 0.0;
            }
        }
    }
    //kronecker_prod at example level:
    void diag_kronecker_product(example& ec1, example& ec2, example& ec)
    {
        //ec <= ec1 X ec2
        copy_example_data(&ec, &ec1);
        ec.total_sum_feat_sq = 0.0;
        for(namespace_index c : ec.indices){
            for(namespace_index c2 : ec2.indices){
                if (c == c2)
                    diag_kronecker_prod_fs(ec.feature_space[c], ec2.feature_space[c2], ec.total_sum_feat_sq);
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

    //void kronecker_product(example& ec1, example& ec2, example& ec, uint64_t mask, size_t ss)
    //{
        //copy_example_data(&ec, &ec1);
    //    ec.indices.push_back(node_id_namespace);
    //    ec.total_sum_feat_sq = 0.0;
    //    for (namespace_index c1 : ec1.indices){
    //        for (namespace_index c2 : ec2.indices){
    //            if (c1 == c2)
    //                kronecker_product_f(ec1.feature_space[c1], ec2.feature_space[c2], ec.feature_space[node_id_namespace], ec.total_sum_feat_sq, ec.num_features, mask, ss);
    //        }
    //    }
    //}

    void kronecker_product(example& ec1, example& ec2, example& ec)
    {
        copy_example_data(&ec, &ec1, true);
        ec.indices.delete_v();
        ec.indices.push_back(node_id_namespace);
        ec.indices.push_back(dictionary_namespace);
        ec.num_features = 0;
        ec.total_sum_feat_sq = 0.0;

        //to do: figure out how to set the indicies correctly (different namespaces may have same index)
        for (auto nc : ec1.indices)
        {
            for (int i = 0; i < ec1.feature_space[nc].indicies.size(); i++){
                ec.feature_space[node_id_namespace].push_back(ec1.feature_space[nc].values[i], ec1.feature_space[nc].indicies[i]);
                ec.num_features++;
                ec.total_sum_feat_sq+=pow(ec1.feature_space[nc].values[i],2);
            }
        }
        for (auto nc : ec2.indices)
        {
            for (int i = 0; i < ec2.feature_space[nc].indicies.size(); i++){
                ec.feature_space[dictionary_namespace].push_back(ec2.feature_space[nc].values[i], ec2.feature_space[nc].indicies[i]);
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
            nl = 0.1; //initilze to 1, as we need to do nl/nr.
            nr = 0.1; 
            examples_index = {};
        }
    };

    //memory_tree
    struct memory_tree
    {
        vw* all;

        //std::vector<int> labels;
        size_t unique_labels;

        //std::vector<node> nodes;
        v_array<node> nodes;  //array of nodes.
        v_array<example*> examples; //array of example points
        //v_array<flat_example*> flat_examples;
        
        size_t max_leaf_examples; 
        size_t max_depth;
        size_t max_routers;
        float alpha; //for cpt type of update.
        size_t valid_nodes_num;
        int iter;
        
        uint32_t num_mistakes;
        uint32_t num_ecs;
        uint32_t num_test_ecs;
        uint32_t test_mistakes;

        memory_tree()
        {
            nodes = {};
            examples = {};
            alpha = 0.5;
            valid_nodes_num = 0;
            iter = 0;
            num_mistakes = 0;
            num_ecs = 0;
            test_mistakes = 0;
            num_test_ecs = 0;
        }
    };

    void init_tree(memory_tree& b, uint32_t root,
            uint32_t depth, uint32_t& routers_used)
    {
        if(depth <= b.max_depth)
        {
            uint32_t left_child;
            uint32_t right_child;
            left_child = (uint32_t)b.nodes.size(); //the next available index in the vector
            b.nodes.push_back(node());
            right_child = (uint32_t)b.nodes.size(); 
            b.nodes.push_back(node());
            b.nodes[root].base_router = routers_used++;
            
            //b.nodes[root].internal = true;
            b.nodes[root].left = left_child;
            b.nodes[left_child].parent = root;
            b.nodes[left_child].depth = depth;
            b.nodes[root].right = right_child;
            b.nodes[right_child].parent = root;
            b.nodes[right_child].depth = depth;

            init_tree(b, left_child, depth+1, routers_used);
            init_tree(b, right_child, depth+1, routers_used);
        }
    }

    
    void init_tree(memory_tree& b)
    {
        //simple initilization: initilize the root only
        uint32_t routers_used = 0;
        b.nodes.push_back(node());
        b.nodes[0].internal = -1; //mark the root as leaf
        b.valid_nodes_num++;
        init_tree(b, 0, 1, routers_used);
        b.max_routers = routers_used;

        cout<<"tree initiazliation is done...."<<endl
            <<"Depth "<<b.max_depth<<endl
            <<"Routers "<<routers_used<<endl
            <<"valid nodes num "<<b.valid_nodes_num<<endl
            <<"tree size: "<<b.nodes.size()<<endl;
    }

    //rout based on the prediction
    inline uint32_t descent(node& n, const float prediction)
    {
        //prediction <0 go left, otherwise go right
        if(prediction < 0)
        {
            n.nl++; //increment the number of examples routed to the left.
            return n.left;
        }
        else //otherwise go right.
        {
            n.nr++; //increment the number of examples routed to the right.
            return n.right; 
        }
    }

    //return the id of the example and the leaf id (stored in cn)
    inline int random_sample_example(memory_tree& b, uint32_t& cn)
    {
        int iter = 0;
        for (iter = 0; iter < 100; iter++){
            cn = 0; //always start from the root:
            while (b.nodes[cn].internal == 1)
            {
                float pred = ((double)rand()/(double)RAND_MAX) < (b.nodes[cn].nl*1./(b.nodes[cn].nr+b.nodes[cn].nl)) ? -1.f : 1.f;
                if (pred < 0)
                    cn = b.nodes[cn].left; 
                else
                    cn = b.nodes[cn].right;
            }
            if (b.nodes[cn].examples_index.size() > 0)  //find a leaf who has non-zero many examples
                break;
        }
        if (iter == 100)
            cout<<"ERROR: keep getting to leafs with no examples....."<<endl;
            return 0;

        int loc_at_leaf = rand() % b.nodes[cn].examples_index.size(); //[0, size-1]
        return loc_at_leaf;
    }


    //train the node with id cn, using the statistics stored in the node to
    //formulate a binary classificaiton example.
    float train_node(memory_tree& b, base_learner& base, example& ec, const uint32_t cn)
    {
        //predict, learn and predict
        //note: here we first train the router and then predict.
        MULTICLASS::label_t mc = ec.l.multi;
        uint32_t save_multi_pred = ec.pred.multiclass;

        base.predict(ec, b.nodes[cn].base_router);
        float prediction = ec.pred.scalar; 
        float imp_weight = 1.; //no importance weight.
        double weighted_value = (1.-b.alpha)*log(b.nodes[cn].nl/b.nodes[cn].nr)+b.alpha*prediction;
        float route_label = weighted_value < 0. ? -1.f : 1.f;
        
        ec.l.simple = {route_label, imp_weight, 0.};
        base.learn(ec, b.nodes[cn].base_router); //update the router according to the new example.
        
        base.predict(ec, b.nodes[cn].base_router);
        float save_binary_scalar = ec.pred.scalar;
        
        ec.l.multi = mc;
        ec.pred.multiclass = save_multi_pred;

        return save_binary_scalar;
    }

    
    //turn a leaf into an internal node, and create two children
    //when the number of examples is too big
    void split_leaf(memory_tree& b, base_learner& base, const uint32_t cn)
    {
           
        b.nodes[cn].internal = 1; //swith to internal node.
        uint32_t left_child = b.nodes[cn].left;
        b.nodes[left_child].internal = -1; //switch to leaf
        uint32_t right_child = b.nodes[cn].right;
        b.nodes[right_child].internal = -1; //swith to leaf
        b.valid_nodes_num+=2; 

        //rout the examples stored in the node to the left and right
        for(size_t ec_id = 0; ec_id < b.nodes[cn].examples_index.size(); ec_id++) //scan all examples stored in the cn
        {
            uint32_t ec_pos = b.nodes[cn].examples_index[ec_id];
            MULTICLASS::label_t mc = b.examples[ec_pos]->l.multi;
            uint32_t save_multi_pred = b.examples[ec_pos]->pred.multiclass;

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
            b.examples[ec_pos]->l.multi = mc;
            b.examples[ec_pos]->pred.multiclass = save_multi_pred;
        }
        b.nodes[cn].examples_index.delete_v(); //empty the cn's example list
        b.nodes[cn].nl = std::max(double(b.nodes[left_child].examples_index.size()), 0.01); //avoid to set nl to zero
        b.nodes[cn].nr = std::max(double(b.nodes[right_child].examples_index.size()), 0.01); //avoid to set nr to zero
    }
    
    //pick up the "closest" example in the leaf using the score function.
    int64_t pick_nearest(memory_tree& b, base_learner& base, const uint32_t cn, example& ec)
    {
        if (b.nodes[cn].examples_index.size() > 0)
        {
            float max_score = -FLT_MAX;
            int64_t max_pos = -1;

            for(int i = 0; i < b.nodes[cn].examples_index.size(); i++)
            {
                uint32_t loc = b.nodes[cn].examples_index[i];
                example* kprod_ec = &calloc_or_throw<example>();
                diag_kronecker_product(ec, *b.examples[loc], *kprod_ec);
                base.predict(*kprod_ec, b.max_routers);
                float score = kprod_ec->pred.scalar;
                if (score > max_score){
                    max_score = score;
                    max_pos = (int64_t)loc;
                }
                free(kprod_ec);
            }
            return max_pos;
        }
        else
            return -1;
    }

    void learn_similarity_at_leaf(memory_tree& b, base_learner& base, const uint32_t cn, example& ec)
    {
        for (uint32_t loc : b.nodes[cn].examples_index)
        {
            example* ec_loc = b.examples[loc];
            example* kprod_ec = &calloc_or_throw<example>();
            diag_kronecker_product(ec, *ec_loc, *kprod_ec);
            if (ec.l.multi.label == b.examples[loc]->l.multi.label) //reward = 1:    
                kprod_ec->l.simple = {1., 1., 0.};
            else
                kprod_ec->l.simple = {-1., 1., 0.}; //reward = 0:
            base.learn(*kprod_ec, b.max_routers);

            free(kprod_ec);
        }
    }

    void predict(memory_tree& b, base_learner& base, example& ec)
    {
        MULTICLASS::label_t mc = ec.l.multi;
        uint32_t save_multi_pred = ec.pred.multiclass;
        uint32_t cn = 0;
        while(b.nodes[cn].internal == 1) //if it's internal
        {
            base.predict(ec, b.nodes[cn].base_router);
            uint32_t newcn = ec.pred.scalar < 0 ? b.nodes[cn].left : b.nodes[cn].right; //do not need to increment nl and nr.
            cn = newcn;
        }
        ec.l.multi = mc;
        ec.pred.multiclass = save_multi_pred;

        uint32_t closest_ec = pick_nearest(b, base, cn, ec);
        if (closest_ec != -1){
            ec.pred.multiclass = b.examples[closest_ec]->l.multi.label;
            if(ec.pred.multiclass != ec.l.multi.label)
            {
                ec.loss = ec.weight; //weight in default is 1.
                b.num_mistakes++;
            }
        }
    }

    //node here the ec is already stored in the b.examples, the task here is to rout it to the leaf, 
    //and insert the ec_array_index to the leaf.
    void insert_example(memory_tree& b, base_learner& base, const uint32_t& ec_array_index)
    {
        uint32_t cn = 0; //start from the root.
        while(b.nodes[cn].internal == 1) //if it's internal node:
        {   
            //predict and train the node at cn.
            float router_pred = train_node(b, base, *b.examples[ec_array_index], cn); 
            uint32_t newcn = descent(b.nodes[cn], router_pred); //updated nr or nl
            cn = newcn; 
        }
        if(b.nodes[cn].internal == -1) //get to leaf:
        {
            //insert the example's location (size - 1) to the leaf node:
            b.nodes[cn].examples_index.push_back(ec_array_index);
            float leaf_pred = train_node(b, base, *b.examples[ec_array_index], cn); //tain the leaf as well.
            descent(b.nodes[cn], leaf_pred); //this is a faked descent, the purpose is only to update nl and nr of cn

            //if the number of examples exceeds the max_leaf_examples, and it hasn't reached the depth limit yet, we split:
            if((b.nodes[cn].examples_index.size() >= b.max_leaf_examples) && (b.nodes[cn].depth < b.max_depth))
                split_leaf(b, base, cn); 
            else
                learn_similarity_at_leaf(b, base, cn, *b.examples[ec_array_index]);  //learn similarity function at leaf
        }
    }
    void experience_replay(memory_tree& b, base_learner& base)
    {
        uint32_t cn = 0; //start from root, randomly descent down! 
        int loc_at_leaf = random_sample_example(b, cn);
        uint32_t ec_id = b.nodes[cn].examples_index[loc_at_leaf]; //ec_id is the postion of the sampled example in b.examples. 

        //pop the sampled example, trace back to the root
        pop_at_index(b.nodes[cn].examples_index, loc_at_leaf); //delete ec_id from examples_index
        while (cn != 0) //trace back to the root
        {
            uint32_t parent = b.nodes[cn].parent;
            if (b.nodes[parent].left == cn)
                b.nodes[parent].nl--;
            else if (b.nodes[parent].right == cn)
                b.nodes[parent].nr--;
            
            cn = parent;
        }
        //re-insert:note that we do not have to 
        //re-store the example into b.examples, as it's alreay there
        insert_example(b, base, ec_id); 
    }


    //learn: descent the example from the root while generating binary training
    //example for each node, including the leaf, and store the example at the leaf.
    void learn(memory_tree& b, base_learner& base, example& ec)
    {
        
        b.iter++;
        if(b.iter % 1000 == 0){
            cout<<"At iter "<<b.iter<<", the prediction error "<<b.num_mistakes*1./(b.iter)<<endl;
            //cout<<"At iter "<<b.iter<<", we have total "<<b.num_ecs<<" examples, and total "<<b.labels.size()<<" unique labels, the ratio is "<<b.labels.size()*1./b.num_ecs<<endl;
        }

        example* new_ec = &calloc_or_throw<example>();
        copy_example_data(new_ec, &ec);
        remove_repeat_features_in_ec(*new_ec);
        b.examples.push_back(new_ec);
        b.num_ecs++; 
        
        predict(b, base, *b.examples[b.num_ecs - 1]);   //prediction is stored in ec.pred.multiclass.

        insert_example(b, base, b.num_ecs-1);

        if ((b.iter % 100) == 0)
            experience_replay(b, base);
    }


    void finish(memory_tree& b)
    {
        for (size_t i = 0; i < b.nodes.size(); ++i)
            b.nodes[i].examples_index.delete_v();
        b.nodes.delete_v();
        for (size_t i = 0; i < b.examples.size(); i++)
            free(b.examples[i]);
        b.examples.delete_v();
    }

} //namespace


base_learner* memory_tree_setup(vw& all)
{
    using namespace memory_tree_ns;
    if (missing_option(all, true, "memory_tree", "Use online tree for nearest neighbor search"))
        return nullptr;
    
    new_options(all, "memory tree options")
    ("max_leaf_examples", po::value<uint32_t>()->default_value(10), "maximum number of examples per leaf in the tree")
    ("Alpha", po::value<float>()->default_value(0.5), "Alpha")
    ("max_depth", po::value<uint32_t>()->default_value(10), "maximum depth of the tree");
    add_options(all);

    po::variables_map& vm = all.vm;

    memory_tree& tree = calloc_or_throw<memory_tree> ();
    tree.all = &all;
    tree.max_leaf_examples = vm["max_leaf_examples"].as<uint32_t>();
    tree.max_depth = vm["max_depth"].as<uint32_t>();
    tree.alpha = vm["Alpha"].as<float>();

    init_tree(tree);

    if (! all.quiet)
        all.trace_message << "memory_tree:" << " "
                    <<"max_depth = "<< tree.max_depth << " " 
                    <<"max_leaf_examples = "<<tree.max_leaf_examples<<" "
                    <<"Alpha = "<<tree.alpha
                    <<std::endl;
    
    learner<memory_tree>& l = 
        init_multiclass_learner (&tree, 
                setup_base (all),
                learn,
                predict,
                all.p, 
                tree.max_routers + 1);
    
    srand(time(0));
    l.set_finish(finish);

    return make_base (l);
}


/*
int main(void)
{
    using namespace memory_tree_ns;
        
    memory_tree& tree = calloc_or_throw<memory_tree> ();
    tree.max_depth = 3;
    init_tree(tree);

    std::cout<<tree.nodes.size()<<std::endl;

    //tree.max_depth = 5;
    //init_tree(tree);
    //std::cout<<tree.nodes.size()<<std::endl;
}*/


/*
if (b.num_ecs == 10){
            example* prod_cor = &calloc_or_throw<example>();
            example* prod_incor = &calloc_or_throw<example>();
            example* sub_ec = &calloc_or_throw<example>();
            diag_kronecker_product(*b.examples[b.examples.size()-1], *b.examples[0], *prod_cor);
            diag_kronecker_product(*b.examples[b.examples.size()-1], *b.examples[1], *prod_incor);
            subtract_two_examples(*prod_cor, *prod_incor, sub_ec);

            cout<<"ab"<<endl;
        }


    void test_substract_features()
    {
        features& f1 = calloc_or_throw<features>();
        features& f2 = calloc_or_throw<features>();
        features& f = calloc_or_throw<features>();
        f1.push_back(1, 1);
        f1.push_back(1, 2);
        f2.push_back(1, 2);
        f2.push_back(0., 3);

        subtract_features(f1,f2,f);
        for (int i = 0 ; i < f.values.size(); i++)
            cout<<f.values[i]<<endl;

    }


float linear_kernel(const flat_example* fec1, const flat_example* fec2)
    { 
        float dotprod = 0;
        features& fs_1 = (features&)fec1->fs;
        features& fs_2 = (features&)fec2->fs;
        if (fs_2.indicies.size() == 0)
            return 0.f;
  
        int numint = 0;
        for (size_t idx1 = 0, idx2 = 0; idx1 < fs_1.size() && idx2 < fs_2.size() ; idx1++)
        { 
            uint64_t ec1pos = fs_1.indicies[idx1];
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

    float normalized_dotprod(memory_tree& b, example& ec1, example& ec2)
    {
        flat_example* fec1 = flatten_sort_example(*(b.all), &ec1);
        flat_example* fec2 = flatten_sort_example(*(b.all), &ec2);
        float dot_prod = linear_kernel(fec1, fec2);
        float normalized_dot_prod = dot_prod / (pow(fec1->total_sum_feat_sq,0.5) * pow(fec2->total_sum_feat_sq,0.5));
        //float square_l2_dis = fec1->total_sum_feat_sq + fec2->total_sum_feat_sq - 2.*dot_prod;

        free(fec1); //free the tempory constructed fec1.
        free(fec2);
        //return square_l2_dis;
        return -normalized_dot_prod;
    }




*/