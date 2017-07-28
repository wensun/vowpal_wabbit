#include <algorithm>
#include <cmath>
#include <cstdio>
#include <float.h>
#include <sstream>
#include "vw.h"

#include "reductions.h"
#include "rand48.h"

using namespace std;
using namespace LEARNER;

namespace log_multi_label_tree_nc{

struct top_F_node_pred{
    uint32_t label; //label belongs to [1,..K]
    float avg_rew;  
    top_F_node_pred(){ label = 0; avg_rew = 0.;}
    top_F_node_pred(uint32_t a): label(a), avg_rew(0.){}
};


struct node{
    uint32_t parent;
    bool internal;
    uint32_t depth;
    uint32_t F;  
    uint32_t k;
    uint32_t base_router;
    uint32_t left;
    uint32_t right;
    float n; //number of examples routed to this node so far.
    float num_all_zeros; //number of examples who has no positive labels in the top_F.

    v_array<float> label_histogram;
    v_array<top_F_node_pred> top_F_label_info;

    node(){
        F = 0;
        k = 0;
        parent = 0;
        internal = false;
        depth = 0;
        base_router = 0;
        left = 0;
        right = 0;
        n = 0.001;
        num_all_zeros = 0.001;
        label_histogram  = v_init<float>();
        top_F_label_info = v_init<top_F_node_pred>();
    }

    node(uint32_t k_, uint32_t F_){
        F = F_;
        k = k_;
        parent = 0;
        internal = false;
        depth = 0;
        base_router = 0;
        left = 0;
        right = 0;
        n = 0.001;
        num_all_zeros = 0.001;
        label_histogram  = v_init<float>();
        for (float i = 0.; i < k; i++)
            label_histogram.push_back(0); //<1, 2, ..., K>
        top_F_label_info = v_init<top_F_node_pred>();
        for (uint32_t i = 1; i <= F; i++){
            top_F_label_info.push_back(top_F_node_pred(i));
        }
        
    }
};

struct log_multi_label_tree {
    vw* all;
    uint32_t k;    
    v_array<node> nodes;
    size_t F;
    size_t max_routers; 
    size_t max_depth;
    uint32_t iter;
    uint32_t mistakes; 

    log_multi_label_tree(){
        k = 0;
        nodes = v_init<node>();
        F = 0;
        max_routers = 0;
        max_depth = 0;
        mistakes = 0;
        iter = 0;
    }
};

void init_tree(log_multi_label_tree& b, uint32_t root, uint32_t depth, uint32_t& routers_used){
    if (depth <= b.max_depth){
        uint32_t left_child;
        uint32_t right_child;
        left_child = (uint32_t)b.nodes.size();
        b.nodes.push_back(node(b.k, b.F));
        right_child = (uint32_t)b.nodes.size();
        b.nodes.push_back(node(b.k, b.F));
        b.nodes[root].base_router = routers_used++;

        b.nodes[root].internal = true;
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

void init_tree (log_multi_label_tree& b){
    uint32_t routers_used = 0;
    b.nodes.push_back(node(b.k, b.F));
    init_tree (b, 0, 1, routers_used);
    b.max_routers = routers_used;

    std::cout<<"finished initializing tree..."<<endl;

}

//computes the probability that there exist at least one label with rew = 1.
inline float statistics_of_top_F(float num_all_zeros, float total_num){
    float prob_all_zeros = num_all_zeros / total_num;
    return log(1. - prob_all_zeros + 0.00000001)/log(2);
}

inline size_t find_min_in_top_F(const node& cnode){
    size_t min_loc = 0;
    float min_v = FLT_MAX;
    for (size_t i = 0; i < cnode.top_F_label_info.size(); i++){
        if (cnode.top_F_label_info[i].avg_rew <= min_v){
            min_v = cnode.top_F_label_info[i].avg_rew;
            min_loc = i;
        }
    }
    return min_loc; //location in the v_array. 
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

inline bool in_top_F(const node& cnode, uint32_t label, size_t& pos){
    pos = 0;
    for (size_t i = 0; i < cnode.top_F_label_info.size(); i++){
        if (cnode.top_F_label_info[i].label == label){
            pos = i;
            return true;
        }
    }
    return false;
}

void update_node(node& cnode, example& ec){
    v_array<uint32_t>& ec_labels = ec.l.multilabels.label_v;
    //update node
    cnode.n ++; 
    for (uint32_t lab : ec_labels)
        cnode.label_histogram[lab-1]++;  //this is because label starts from 1, ends at K.

    //update top_F information:
    for (size_t i = 0; i < cnode.top_F_label_info.size(); i++){
        uint32_t lab = cnode.top_F_label_info[i].label;
        cnode.top_F_label_info[i].avg_rew = cnode.label_histogram[lab-1]*1./cnode.n;
    }

    for (uint32_t label : ec_labels){
        size_t label_pos_in_top_F = 0;
        bool in = in_top_F(cnode, label, label_pos_in_top_F);
        if (in == true)
            cnode.top_F_label_info[label_pos_in_top_F].avg_rew = cnode.label_histogram[label-1]*1./cnode.n;
        else{
            size_t min_loc = find_min_in_top_F(cnode);
            float avg_rew_min_loc = cnode.top_F_label_info[min_loc].avg_rew;
            if (cnode.label_histogram[label-1]*1./cnode.n > avg_rew_min_loc){
                cnode.top_F_label_info[min_loc].label = label;
                cnode.top_F_label_info[min_loc].avg_rew = cnode.label_histogram[label-1]*1./cnode.n;
            }
        }
    }
    bool overlap_with_top_F = false;
    for (uint32_t label: ec_labels){
        size_t tmp_loc = 0;
        bool in = in_top_F(cnode, label, tmp_loc);
        if (in == true){
            overlap_with_top_F = true;
            break;
        }
    }
    if (overlap_with_top_F == false)
        cnode.num_all_zeros++;

}

//to do: implement fake update steps
float fake_update_node(node& cnode, example& ec){ 
    v_array<uint32_t>& labels = ec.l.multilabels.label_v;
    v_array<top_F_node_pred> tmp_nodes = v_init<top_F_node_pred>();
    for(int i = 0; i < cnode.top_F_label_info.size(); i++){
        tmp_nodes.push_back(top_F_node_pred(cnode.top_F_label_info[i].label));
        tmp_nodes[i].avg_rew = cnode.top_F_label_info[i].avg_rew;
    }
    float old_num_all_zeros = cnode.num_all_zeros;
    update_node(cnode, ec);
    float new_num_all_zeros = cnode.num_all_zeros;
    //undo upudate
    cnode.n--;
    for (uint32_t label : labels)
        cnode.label_histogram[label-1]--;
    for (int i = 0; i < cnode.top_F_label_info.size(); i++){
        cnode.top_F_label_info[i].label = tmp_nodes[i].label;
        cnode.top_F_label_info[i].avg_rew = tmp_nodes[i].avg_rew;
    }
    tmp_nodes.delete_v();
    cnode.num_all_zeros = old_num_all_zeros;

    return new_num_all_zeros; 
}

//store ec's multilabel information into tmp_l and tmp_pred, and restore later.
inline void save_restore_multilabel_info(example& ec, 
            v_array<uint32_t>& tmp_l_multilabels,
            v_array<uint32_t>& tmp_pred_multilabels, bool save){
    if (save == true){
        //v_array<uint32_t> tmp_l_multilabels = v_init<uint32_t>;
        //v_array<uint32_t> tmp_pred_multilabels=v_init<uint32_t>;
        for (int i = 0; i < ec.l.multilabels.label_v.size(); i++)
            tmp_l_multilabels.push_back(ec.l.multilabels.label_v[i]);
        
        for (int i = 0; i < ec.pred.multilabels.label_v.size(); i++)
            tmp_pred_multilabels.push_back(ec.pred.multilabels.label_v[i]);
    } 
    else{
        ec.l.multilabels.label_v.delete_v();
        ec.l.multilabels.label_v = v_init<uint32_t>();
        ec.pred.multilabels.label_v.delete_v();
        ec.pred.multilabels.label_v = v_init<uint32_t>();
        
        for (int i = 0; i < tmp_l_multilabels.size(); i++)
            ec.l.multilabels.label_v.push_back(tmp_l_multilabels[i]);
        for (int i = 0; i < tmp_pred_multilabels.size(); i++)
            ec.pred.multilabels.label_v.push_back(tmp_pred_multilabels[i]);
        tmp_l_multilabels.delete_v();
        tmp_pred_multilabels.delete_v();
    }           
}


void train_internal_node(log_multi_label_tree&b, base_learner& base, uint32_t cn, example& ec){
    if (b.nodes[cn].internal == false){
        cout<<"Error: try to train a leaf node..."<<endl;
        exit(0);
    }
    MULTILABEL::labels multilabels = ec.l.multilabels;
    MULTILABEL::labels preds = ec.pred.multilabels;

    //v_array<uint32_t> tmp_l_multilabels = v_init<uint32_t>();
    //v_array<uint32_t> tmp_pred_multilabels=v_init<uint32_t>();
    //save_restore_multilabel_info(ec, tmp_l_multilabels, tmp_pred_multilabels, true);

    uint32_t left_child = b.nodes[cn].left;
    float nl = b.nodes[left_child].n;
    uint32_t right_child = b.nodes[cn].right;
    float nr = b.nodes[right_child].n;

    float all_zeros_left = b.nodes[left_child].num_all_zeros;
    float all_zeros_right = b.nodes[right_child].num_all_zeros;

    float all_zeros_left_p = fake_update_node(b.nodes[left_child], ec);
    float all_zeros_right_p= fake_update_node(b.nodes[right_child],ec);

    float benefit_left = ((nl+1.)/(nl+nr+1.)*statistics_of_top_F(all_zeros_left_p,nl+1.)
                        + nr/(nl+nr+1.)*statistics_of_top_F(all_zeros_right, nr));
    float benefit_right= ((nr+1.)/(nl+nr+1.)*statistics_of_top_F(all_zeros_right_p, nr+1.)
                        + nl/(nr+nl+1.)*statistics_of_top_F(all_zeros_left, nl));
    
    float route_label = (benefit_left <= benefit_right ? 1.f : -1.f);
    float weight = fabs((float)(benefit_left - benefit_right));
    ec.l.simple = {route_label, weight, 0.};
    base.learn(ec, b.nodes[cn].base_router);

    //restore:
    ec.pred.multilabels = preds;
    ec.l.multilabels = multilabels;
    //save_restore_multilabel_info(ec, tmp_l_multilabels, tmp_pred_multilabels, false);
}


void add_node_id_feature (log_multi_label_tree& b, uint32_t cn, example& ec){
    vw* all = b.all;
    uint64_t mask = all->weights.mask();
    size_t ss = all->weights.stride_shift();

    ec.indices.push_back (node_id_namespace);
    features& fs = ec.feature_space[node_id_namespace];
    while (cn > 0){ 
        fs.push_back (1., ((868771 * cn) << ss) & mask);
        cn = b.nodes[cn].parent;
    }
}
void remove_node_id_feature (log_multi_label_tree& b, uint32_t cn, example& ec){
    features& fs = ec.feature_space[node_id_namespace];
    fs.erase ();
    ec.indices.pop ();
}


void train_at_leaf(log_multi_label_tree& b, base_learner& base, const uint32_t cn, example& ec){

    MULTILABEL::labels multilabels = ec.l.multilabels;
    MULTILABEL::labels preds = ec.pred.multilabels;

    add_node_id_feature(b, cn, ec);
    v_array<uint32_t>& ec_labels = ec.l.multilabels.label_v;
    v_array<top_F_node_pred>& top_f_nodes = b.nodes[cn].top_F_label_info;
    for (auto label_info : top_f_nodes){
        uint32_t label = label_info.label;
        uint32_t pos = 0;
        bool in = in_v_array(ec_labels, label, pos);
        if (in == true){ //label is in the current example's label list:
            ec.l.simple = {1.f, 1.f, 0.f};
            base.learn (ec, b.max_routers + label - 1);
        }
        else{
            ec.l.simple = {-1.f, 1.f, 0.f};
            base.learn(ec, b.max_routers + label -1);
        }
        ec.l.multilabels = multilabels;
        //ec.l.simple = simple_label;
    }
    remove_node_id_feature(b,cn,ec);
    ec.pred.multilabels = preds;
    ec.l.multilabels = multilabels;
    //save_restore_multilabel_info(ec, tmp_l_multilabels, tmp_pred_multilabels, false);
}


uint32_t routing(log_multi_label_tree& b, base_learner& base, example& ec, bool training)
{
    MULTILABEL::labels multilabels = ec.l.multilabels;
    MULTILABEL::labels preds = ec.pred.multilabels;

    uint32_t cn = 0;
    if (training == true){
        update_node(b.nodes[cn], ec);
    }

    while(b.nodes[cn].internal == true){ //internal node:
        if (training == true){
            train_internal_node(b, base, cn, ec);
        }
        
        
        ec.l.simple.label = FLT_MAX;
        base.predict(ec, b.nodes[cn].base_router);
        ec.l.multilabels = multilabels;
        
        float pred_scalar = ec.pred.scalar;

        uint32_t newcn = (pred_scalar < 0 ? b.nodes[cn].left : b.nodes[cn].right);

        if (training)
            update_node(b.nodes[newcn], ec);
        cn = newcn;
    }
    ec.pred.multilabels = preds;
    ec.l.multilabels = multilabels;

    return cn;
}

void predict(log_multi_label_tree& b, base_learner& base, example& ec){
    //v_array<uint32_t> tmp_l_multilabels = v_init<uint32_t>();
    //v_array<uint32_t> tmp_pred_multilabels=v_init<uint32_t>();
    //save_restore_multilabel_info(ec, tmp_l_multilabels, tmp_pred_multilabels, true);
    
    MULTILABEL::labels multilabels = ec.l.multilabels;
    MULTILABEL::labels preds = ec.pred.multilabels;
    uint32_t node_id = routing(b, base, ec, false);
    //temporaliy compute P@1:

    ec.l.simple = {1.f, 0.f, 0.0};
    add_node_id_feature (b, node_id, ec);
    uint32_t max_label = 0;
    float max_pred = -FLT_MAX;

    ec.l.simple.label = FLT_MAX;
    for (auto top_f_lab : b.nodes[node_id].top_F_label_info){
        base.predict(ec, b.max_routers + top_f_lab.label - 1);
        if (max_pred < ec.partial_prediction || max_label == 0){
            max_pred = ec.partial_prediction;
            max_label = top_f_lab.label;
        }
    }
    //ec.l.simple = simple_lab;
    ec.l.multilabels = multilabels;
    ec.pred.multiclass = max_label;
    preds.label_v.erase();
    preds.label_v.push_back(max_label);
    ec.pred.multilabels = preds;


    remove_node_id_feature (b, node_id, ec);   
    
    uint32_t pos = 0;
    bool in = in_v_array(ec.l.multilabels.label_v, max_label, pos);
    if (in == false){
        //ec.loss = 1.;
        b.mistakes++;
    }

}

void learn(log_multi_label_tree& b, base_learner& base, example& ec){
    predict(b, base, ec);
    uint32_t node_id = 0;
    node_id = routing(b, base, ec, true);
    train_at_leaf(b, base, node_id, ec);
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
{ MULTILABEL::labels& ld = ec.l.multilabels;
  float loss = 0.;
  if (!is_test_label(ld))
  { //need to compute exact loss
    
    uint32_t pos = 0;
    bool in = in_v_array(ec.l.multilabels.label_v, ec.pred.multilabels.label_v[0], pos);
    if (in == false){
        loss ++;
    }
    
    /*uint32_t preds_index = 0;
    uint32_t given_index = 0;
    while(preds_index < preds.label_v.size() && given_index < given.label_v.size())
    { if (preds.label_v[preds_index] < given.label_v[given_index])
        preds_index++;
      else if (preds.label_v[preds_index] > given.label_v[given_index])
      { given_index++;
        loss++;
      }
      else
      { preds_index++;
        given_index++;
      }
    }
    loss += given.label_v.size() - given_index;
    loss += preds.label_v.size() - preds_index;*/
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




void finish_example(vw& all, log_multi_label_tree&, example& ec)
{
    output_example(all, ec);
    VW::finish_example(all, &ec);
}


void finish(log_multi_label_tree& b){
    for (size_t i = 0; i < b.nodes.size(); i++){
        b.nodes[i].label_histogram.delete_v();
        b.nodes[i].top_F_label_info.delete_v();
    }
    b.nodes.delete_v();
    cout<<"done clearning.."<<endl;
}


////////////////////Save & Load stuff//////////////////////
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


void save_load_node(node& cn, io_buf& model_file, bool& read, bool& text, stringstream& msg){
    writeit(cn.parent, "parent");
    writeit(cn.internal, "internal");
    writeit(cn.depth, "depth");
    writeit(cn.base_router, "base_router");
    writeit(cn.left, "left");
    writeit(cn.right, "right");
    writeit(cn.n, "n");    
    writeit(cn.num_all_zeros, "num_all_zeros");
    writeit(cn.F, "F");
    writeit(cn.k, "node_k");
    writeitvar(cn.label_histogram.size(), "histogram_length", hist_len);
    writeitvar(cn.top_F_label_info.size(), "top_F_len", top_F_len);

    if (read){
        cn.label_histogram.erase();
        for (uint32_t k = 0; k < hist_len; k++)
            cn.label_histogram.push_back(0);
    }
    for (uint32_t k = 0; k < hist_len; k++)
        writeit(cn.label_histogram[k], "histogram_label");
    
    if (read){
        cn.top_F_label_info.erase();
        for (uint32_t k = 0; k < top_F_len; k++)
            cn.top_F_label_info.push_back(top_F_node_pred());
    }
    for (uint32_t k = 0; k < top_F_len; k++){
        writeit(cn.top_F_label_info[k].label, "top_F_label");
        writeit(cn.top_F_label_info[k].avg_rew, "top_F_avg_rew");
    }
}


void save_load_tree(log_multi_label_tree& b, io_buf& model_file, bool read, bool text){
    if (model_file.files.size() > 0)
    {
        stringstream msg;
        writeit(b.k, "k");
        writeit(b.F, "max_candidates");
        writeit(b.max_depth, "max_depth");
        writeitvar(b.nodes.size(), "nodes", n_nodes);
        if (read){
            b.nodes.erase();
            for (uint32_t j =0; j< n_nodes; j++)
                b.nodes.push_back(node());
        }
        for (uint32_t i = 0; i < n_nodes; i++)
            save_load_node(b.nodes[i], model_file, read, text, msg);  
    }
}


} //namespace

base_learner* log_time_multi_label_tree_setup(vw& all)
{
    using namespace log_multi_label_tree_nc;
    if (missing_option<size_t, true>(all,
                                   "log_time_multi_label_tree",
                                   "Use online tree for extreme multilabel classification with <k> classes"))
        return nullptr;

    new_options(all, "log time multilabel tree options")
    ("max_candidates", po::value<uint32_t>(), "maximum number of labels per leaf in the tree")
    ("max_depth", po::value<uint32_t>(), "maximum depth of the tree, default log_2 (#classes)");
    add_options(all);

    po::variables_map& vm = all.vm;
    
    log_multi_label_tree& tree = calloc_or_throw<log_multi_label_tree>();
    tree.all = &all;
    tree.k = (uint32_t)vm["log_time_multi_label_tree"].as<size_t>();
    tree.F = vm.count("max_candidates") > 0
        ? vm["max_candidates"].as<uint32_t>()
        : (std::min)(tree.k, 5*(uint32_t)(ceil(log(tree.k)/log(2.))));
    *(all.file_options) << " --max_candidates " << tree.F;

    tree.max_depth = 
        vm.count("max_depth") > 0
        ? vm["max_depth"].as<uint32_t>()
        : (uint32_t)(ceil(log(tree.k)));
    *(all.file_options) << " --max_depth " << tree.max_depth;

    init_tree(tree);

    if (! all.quiet)
        all.trace_message << "log_time_multi_label_tree:"
              << " max_depth = " << tree.max_depth
              << " max_candidates = "<<tree.F
              << std::endl;

    //learner<log_multi_label_tree>& l = 
    //    init_multiclass_learner(&tree, setup_base(all), 
    //                learn, 
    //                predict,
    //                all.p,
    //                tree.max_routers + tree.k);

    learner<log_multi_label_tree>& l = 
        init_learner(&tree, setup_base(all), learn, predict, tree.max_routers+tree.k,
                    prediction_type::multilabels);
    
    l.set_save_load(save_load_tree);
    l.set_finish_example(finish_example);
    l.set_finish(finish); 
    all.p->lp = MULTILABEL::multilabel;
    all.label_type = label_type::multi;
    all.delete_prediction = MULTILABEL::multilabel.delete_label;
    return make_base(l);
}