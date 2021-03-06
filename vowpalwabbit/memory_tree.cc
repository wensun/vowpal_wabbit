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


using namespace std;
using namespace LEARNER;

namespace memory_tree_ns
{
    ///////////////////////Helper//////////////////////////////
    //////////////////////////////////////////////////////////
    //
    float collision_cleanup(features& fs)
    {
  	uint64_t last_index = (uint64_t)-1;
  	float sum_sq = 0.f;
  	features::iterator pos = fs.begin();
  	for (features::iterator& f : fs)
  	{
    		if (last_index == f.index())
      			pos.value() += f.value();
    		else
    		{
      			sum_sq += pos.value() * pos.value();
      			++pos;
      			pos.value() = f.value();
      			pos.index() = f.index();
      			last_index = f.index();
    		}
  	}

  	sum_sq += pos.value() * pos.value();
 	 fs.sum_feat_sq = sum_sq;
  	++pos;
  	fs.truncate_to(pos);

  	return sum_sq;
    }



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


    ////Implement kronecker_product between two examples:
    //kronecker_prod at feature level:
    void diag_kronecker_prod_fs_test(features& f1, features& f2, features& prod_f, float& total_sum_feat_sq, float norm_sq1, float norm_sq2)
    {
        prod_f.delete_v();
        if (f2.indicies.size() == 0)
            return;
    
        for (size_t idx1 = 0, idx2 = 0; idx1 < f1.size() && idx2 < f2.size(); idx1++)
        {
            uint64_t ec1pos = f1.indicies[idx1];
            uint64_t ec2pos = f2.indicies[idx2];
            if (ec1pos < ec2pos) continue;

            while (ec1pos > ec2pos && ++idx2 < f2.size())
                ec2pos = f2.indicies[idx2];

            if (ec1pos == ec2pos){
                prod_f.push_back(f1.values[idx1]*f2.values[idx2]/pow(norm_sq1*norm_sq2,0.5), ec1pos);
                total_sum_feat_sq+=f1.values[idx1]*f2.values[idx2]/pow(norm_sq1*norm_sq2,0.5);
                ++idx2;
            }
        }
    }

    void diag_kronecker_product_test(example& ec1, example& ec2, example& ec)
    {
        copy_example_data(&ec, &ec1);
        ec.total_sum_feat_sq = 0.0;
        for(namespace_index c1 : ec1.indices){
            for(namespace_index c2 : ec2.indices){
                if (c1 == c2 && c1 != node_id_namespace) //do not include the predicted router error features.
                    diag_kronecker_prod_fs_test(ec1.feature_space[c1], ec2.feature_space[c2], ec.feature_space[c1], ec.total_sum_feat_sq, ec1.total_sum_feat_sq, ec2.total_sum_feat_sq);
            }
        }
    }


    /*
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
        copy_example_data(&ec, &ec1);
        ec.sorted = false;
        ec.total_sum_feat_sq = 0.0;
        for(namespace_index c : ec.indices){
            for(namespace_index c2 : ec2.indices){
                if (c == c2)
                    diag_kronecker_prod_fs(ec.feature_space[c], ec2.feature_space[c2], ec.total_sum_feat_sq, ec1.total_sum_feat_sq, ec2.total_sum_feat_sq);
            }
        }
    }*/

    /*
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
    */

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
        uint32_t base_router_error; //use to index the predictor for router error
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
            base_router_error = 0;
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

        float construct_time;
        float test_time;

        bool path_id_feat;

        uint32_t num_mistakes;
        uint32_t num_ecs;
        uint32_t num_test_ecs;
        uint32_t test_mistakes;
        bool learn_at_leaf;
	bool router_error_feature;
        bool test_mode;
        

        memory_tree()
        {
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
            construct_time = 0;
            test_time = 0;
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
   
   
   float normalized_linear_prod(memory_tree& b, example* ec1, example* ec2)
    {
        flat_example* fec1 = flatten_sort_example(*b.all, ec1);
        flat_example* fec2 = flatten_sort_example(*b.all, ec2);
        float linear_prod = linear_kernel(fec1, fec2);
        fec1->fs.delete_v(); 
        fec2->fs.delete_v();
        free(fec1);
        free(fec2);
        return linear_prod/pow(fec1->total_sum_feat_sq*fec2->total_sum_feat_sq, 0.5);
    }
    

   float linear_kernel_features(const features& fs_1, const features& fs_2)
   {
       float dotprod = 0;
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
        return dotprod;
   }

   /*
   float normalized_linear_prod(memory_tree& b, example* ec1, example* ec2)
   {
       float linear_prod = 0;
       float ec1_total_sum_feat_sq = 0;
       float ec2_total_sum_feat_sq = 0;
       for(namespace_index c1 : ec1->indices){
            for(namespace_index c2 : ec2->indices){
                if (c1 == c2 && c1 != node_id_namespace){ //do not include the predicted router error features.
		    ec1->feature_space[c1].sort(b.all->parse_mask);
		    ec1_total_sum_feat_sq += collision_cleanup(ec1->feature_space[c1]);
    		    ec2->feature_space[c2].sort(b.all->parse_mask);
		    ec2_total_sum_feat_sq += collision_cleanup(ec2->feature_space[c2]);
                    linear_prod += linear_kernel_features(ec1->feature_space[c1], ec2->feature_space[c2]);
		}
            }
       }
       return linear_prod/pow(ec1_total_sum_feat_sq*ec2_total_sum_feat_sq, 0.5);
   }*/

    void init_tree(memory_tree& b)
    {
        srand48(4000);
        //simple initilization: initilize the root only
        b.routers_used = 0;

	//uint32_t max_num_routers = b.max_nodes + 1;
        //if (b.router_error_feature == true)
        //    max_num_routers *= 2;

        b.nodes.push_back(node());
        b.nodes[0].internal = -1; //mark the root as leaf
        b.nodes[0].base_router = (b.routers_used++);
	b.nodes[0].depth = 0;
        //if (b.router_error_feature == true)
        //    b.nodes[0].base_router_error = (b.routers_used++);

        cout<<"tree initiazliation is done...."<<endl
            <<"max nodes "<<b.max_nodes<<endl
            <<"tree size: "<<b.nodes.size()<<endl
            <<"learn at leaf: "<<b.learn_at_leaf<<endl
	    <<"using predicted router error as feature: "<<b.router_error_feature<<endl;
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

    //add path feature:
    void add_learned_router_error_feature (memory_tree& b, base_learner& base, uint32_t cn, example& ec)
    {
        MULTICLASS::label_t mc = ec.l.multi;
        uint32_t save_pred = ec.pred.multiclass;

        vw* all = b.all;
        uint64_t mask = all->weights.mask();
        size_t ss = all->weights.stride_shift();

	if (b.nodes[cn].depth == 0)
            ec.indices.push_back (node_id_namespace);
        features& fs = ec.feature_space[node_id_namespace];

        //node_only option:
        ec.l.simple = {FLT_MAX, 0.f, 0.f};
        base.predict(ec, b.max_nodes + b.nodes[cn].depth); //base_router_error to predict
	if (ec.pred.scalar > 0.0) //error:
            fs.push_back(1., ((868771 * cn) << ss) & mask); //add predicted router error information to feature

        ec.l.multi = mc;
        ec.pred.multiclass = save_pred;
    }

    void remove_learned_router_error_feature(memory_tree& b, example& ec){
    	features& fs = ec.feature_space[node_id_namespace];
	fs.erase();
	ec.indices.pop();
    }


    //train the node with id cn, using the statistics stored in the node to
    //formulate a binary classificaiton example.
    float train_node(memory_tree& b, base_learner& base, example& ec, const uint32_t cn)
    {
        //predict, learn and predict
        //note: here we first train the router and then predict.
        MULTICLASS::label_t mc = ec.l.multi;
        uint32_t save_multi_pred = ec.pred.multiclass;
        ec.l.simple = {FLT_MAX,0.f, 0.f};
        base.predict(ec, b.nodes[cn].base_router);
        float prediction = ec.pred.scalar; 
        float imp_weight = 1.f; //no importance weight.
        
        double weighted_value = (1.-b.alpha)*log(b.nodes[cn].nl/b.nodes[cn].nr)/log(2.)+b.alpha*prediction;
        float route_label = weighted_value < 0.f ? -1.f : 1.f;
        
	if (b.router_error_feature == true)
	    add_learned_router_error_feature(b,base,cn,ec);

        ec.l.simple = {route_label, imp_weight, 0.f};
	base.learn(ec, b.nodes[cn].base_router); //update the router according to the new example.
	ec.l.simple = {FLT_MAX,0.f, 0.f};
        base.predict(ec, b.nodes[cn].base_router);
        float save_binary_scalar = ec.pred.scalar;

        //train the predictor to predict the router error and add the prediction to features.
        //the goal is to learn a predictor that can predicts the routing error.
        if (b.router_error_feature == true){
            //float router_error = (ec.pred.scalar*route_label) < 0.f ? 1.f : -1.f;   //if router has error, set 1, otherwise set it -1. 
            //ec.l.simple = {router_error, 1., 0.f};
            //base.learn(ec, b.max_nodes+b.nodes[cn].depth+1); //update the predictor using the router_error.
	    
	    //test the idea of using a depth-wise router and adding it's output as the feature
	    ec.l.simple = {route_label, imp_weight, 0.f};
	    base.learn(ec, b.max_nodes+b.nodes[cn].depth);
        }

        ec.l.multi = mc;
        ec.pred.multiclass = save_multi_pred;
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
        if (b.router_error_feature == true)
            b.nodes[left_child].base_router_error = (b.routers_used++);

        uint32_t right_child = (uint32_t)b.nodes.size();
        b.nodes.push_back(node());  
        b.nodes[right_child].internal = -1;  //right leaf
        b.nodes[right_child].base_router = (b.routers_used++); 
        if (b.router_error_feature == true)
            b.nodes[right_child].base_router_error = (b.routers_used++);

        if (b.nodes[cn].depth + 1 > b.max_depth){
            b.max_depth = b.nodes[cn].depth + 1;
            cout<<"depth "<<b.max_depth<<endl;
        }

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
            MULTICLASS::label_t mc = b.examples[ec_pos]->l.multi;
            uint32_t save_multi_pred = b.examples[ec_pos]->pred.multiclass;
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
            b.examples[ec_pos]->l.multi = mc;
            b.examples[ec_pos]->pred.multiclass = save_multi_pred;
        }
        b.nodes[cn].examples_index.delete_v(); //empty the cn's example list
        b.nodes[cn].nl = std::max(double(b.nodes[left_child].examples_index.size()), 0.001); //avoid to set nl to zero
        b.nodes[cn].nr = std::max(double(b.nodes[right_child].examples_index.size()), 0.001); //avoid to set nr to zero

        if (std::max(b.nodes[cn].nl, b.nodes[cn].nr) > b.max_ex_in_leaf)
        {
            b.max_ex_in_leaf = std::max(b.nodes[cn].nl, b.nodes[cn].nr);
            cout<<b.max_ex_in_leaf<<endl;
        }

    }
    
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
                    float tmp_s = normalized_linear_prod(b, &ec, b.examples[loc]);
                    //tmp_s = 0;
                    example* kprod_ec = &calloc_or_throw<example>();
                    diag_kronecker_product_test(ec, *b.examples[loc], *kprod_ec);
                    kprod_ec->l.simple = {FLT_MAX, 1., tmp_s};
                    base.predict(*kprod_ec, b.max_routers);
                    score = kprod_ec->partial_prediction;
                    free_example(kprod_ec);
                }
                else
                    score = normalized_linear_prod(b, &ec, b.examples[loc]);
                
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

    void learn_similarity_at_leaf(memory_tree& b, base_learner& base, const uint32_t cn, example& ec)
    {
        for (uint32_t loc : b.nodes[cn].examples_index)
        {
            example* ec_loc = b.examples[loc];
            //float score = -compute_l2_distance(b, &ec, ec_loc); //score based on the l2_distance. 
            float score = normalized_linear_prod(b, &ec, ec_loc);
            //score = 0.0;
            example* kprod_ec = &calloc_or_throw<example>();
            diag_kronecker_product_test(ec, *ec_loc, *kprod_ec);
            //
            if (ec.l.multi.label == b.examples[loc]->l.multi.label) //reward = 1:    
                kprod_ec->l.simple = {1., 1., -score};
            else
                //kprod_ec->l.simple = {-1., 1., score}; //reward = 0:
                kprod_ec->l.simple = {-0., 1., -score};
        
            base.learn(*kprod_ec, b.max_routers);
            free_example(kprod_ec);
        }
    }

    void predict(memory_tree& b, base_learner& base, example& test_ec)
    {
        example& ec = calloc_or_throw<example>();
        //ec = *b.examples[b.iter];
        copy_example_data(&ec, &test_ec);
        remove_repeat_features_in_ec(ec);
        
        MULTICLASS::label_t mc = ec.l.multi;
        uint32_t save_multi_pred = ec.pred.multiclass;
        uint32_t cn = 0;
        ec.l.simple = {-1.f, 1.f, 0.};
        while(b.nodes[cn].internal == 1) //if it's internal
        {
            if (b.router_error_feature == true) //add predicted routing error into feature
                add_learned_router_error_feature(b, base, cn, ec);

            base.predict(ec, b.nodes[cn].base_router);
            uint32_t newcn = ec.pred.scalar < 0 ? b.nodes[cn].left : b.nodes[cn].right; //do not need to increment nl and nr.
            cn = newcn;
        }
        ec.l.multi = mc; 
        ec.pred.multiclass = save_multi_pred;

        int64_t closest_ec = pick_nearest(b, base, cn, ec);
        if (closest_ec != -1){
            ec.pred.multiclass = b.examples[closest_ec]->l.multi.label;
            test_ec.pred.multiclass = b.examples[closest_ec]->l.multi.label;
            if(test_ec.pred.multiclass != test_ec.l.multi.label)
            {
                test_ec.loss = test_ec.weight; //weight in default is 1.
                b.num_mistakes++;
            }
        }
        else{
            test_ec.loss = test_ec.weight;
            b.num_mistakes++;
        }
        free_example(&ec);
    }

    //node here the ec is already stored in the b.examples, the task here is to rout it to the leaf, 
    //and insert the ec_array_index to the leaf.
    void insert_example(memory_tree& b, base_learner& base, const uint32_t& ec_array_index, bool fake_insert = false)
    {
        uint32_t cn = 0; //start from the root.
        while(b.nodes[cn].internal == 1) //if it's internal node:
        {   
            //predict and train the node at cn.
            float router_pred = train_node(b, base, *b.examples[ec_array_index], cn); //train_node takes care of adding learning router error features, train router, and update error predictor.
            uint32_t newcn = descent(b.nodes[cn], router_pred); //updated nr or nl
            cn = newcn; 
        }
        //at leaf, use available information at leaf to train regressor/classifier at leaf.
        if ((b.nodes[cn].internal == -1) && (b.learn_at_leaf == true))
            learn_similarity_at_leaf(b, base, cn, *b.examples[ec_array_index]); //currently didn't not use learning router-error feature in the leaf. 

        if((b.nodes[cn].internal == -1) && (fake_insert == false)) //get to leaf:
        {   
            b.nodes[cn].examples_index.push_back(ec_array_index);
            if (b.nodes[cn].examples_index.size() > b.max_ex_in_leaf)
            {
                b.max_ex_in_leaf = b.nodes[cn].examples_index.size();
                //cout<<cn<<" "<<b.max_ex_in_leaf<<endl;
            }
            float leaf_pred = train_node(b, base, *b.examples[ec_array_index], cn); //tain the leaf as well.
            descent(b.nodes[cn], leaf_pred); //this is a faked descent, the purpose is only to update nl and nr of cn

            //if the number of examples exceeds the max_leaf_examples, and not reach the max_nodes - 2 yet, we split:
            if((b.nodes[cn].examples_index.size() >= b.max_leaf_examples) && (b.nodes.size() + 2 <= b.max_nodes)){
                split_leaf(b, base, cn); 
            }
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
	    if (b.router_error_feature == true)
	    	remove_learned_router_error_feature(b, *b.examples[ec_id]); //remove previously added features
            insert_example(b, base, ec_id); //re-insert (no)
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

            clock_t begin = clock();
            example* new_ec = &calloc_or_throw<example>();
            copy_example_data(new_ec, &ec);
            remove_repeat_features_in_ec(*new_ec); ////sort unique.
            b.examples.push_back(new_ec);   
            b.num_ecs++; 

            float random_prob = merand48(b.all->random_state);
            if (random_prob < 1.)
                insert_example(b, base, b.examples.size()-1);
            else{
                insert_example(b, base, b.examples.size()-1, true);
                b.num_ecs--;
                free_example(new_ec);
                b.examples.pop();
            }
            //if (b.iter % 1 == 0)
            for (int i = 0; i < 1; i++)
                experience_replay(b, base);
            b.construct_time = double(clock() - begin)/CLOCKS_PER_SEC;   
        }
        else if (b.test_mode == true){
            b.iter++;
            if (b.iter % 5000 == 0)
                cout<<"at iter "<<b.iter<<", pred error: "<<b.num_mistakes*1./b.iter<<endl;
            clock_t begin = clock();
            predict(b, base, ec);
            b.test_time += double(clock() - begin)/CLOCKS_PER_SEC;
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
        cout<<b.construct_time<<" "<<b.test_time<<endl;
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
            writeit(b.router_error_feature, "router_error_feature");
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

base_learner* memory_tree_setup(vw& all)
{
    using namespace memory_tree_ns;
    if (missing_option<uint32_t, true>(all, "memory_tree", "Make a memory tree with at most <n> nodes"))
        return nullptr;
    
    new_options(all, "memory tree options")
      ("leaf_example_multiplier", po::value<uint32_t>()->default_value(1.0), "multiplier on examples per leaf (default = log nodes)")
      ("learn_at_leaf", po::value<bool>()->default_value(true), "whether or not learn at leaf (defualt = True)")
      ("router_error_feature", po::value<bool>()->default_value(false), "whether or not using prediction of routing error as features (defualt = False)")
      ("Alpha", po::value<float>()->default_value(0.1), "Alpha");
     add_options(all);

    po::variables_map& vm = all.vm;
    memory_tree& tree = calloc_or_throw<memory_tree> ();
    tree.all = &all;
    tree.max_nodes = vm["memory_tree"].as<uint32_t>();
    tree.learn_at_leaf = vm["learn_at_leaf"].as<bool>();
    tree.router_error_feature = vm["router_error_feature"].as<bool>();

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
        all.trace_message << "memory_tree:" << " "
                    <<"max_nodes = "<< tree.max_nodes << " " 
                    <<"max_leaf_examples = "<<tree.max_leaf_examples<<" "
                    <<"alpha = "<<tree.alpha
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
