# primero hay que importar graphviz y todas las dependencias

bst = clf_xgb.get_booster()
for importance_type in ('weight', 'gain', 'cover', 'total_gain', 'total_cover'):
    print('%s: ' % importance_type, bst.get_score(importance_type=importance_type))
node_params = {'shape': 'box', ## make the nodes fancy
                'style': 'filled, rounded',
                'fillcolor': '#78cbe'}
leaf_params= {'shape': 'box',
             'style': 'filled',
            'fillcolor': '#e48038'}
## NOTE: num_trees is NOT the number of trees to plot, but the specific tree you want to plot ## The default value is 0, but I'm setting it just to show it in action since it is ## counter-intuitive.
xgb.to_graphviz(clf_xgb,num_trees=0, size="10,10",
condition_node_params=node_params,
leaf_node_params=leaf_params)
## if you want to save the figure...
# graph_data = xgb.to_graphviz(clf_xgb, num_trees=0, size="10,10",
#condition_node_params=node_params, leaf_node_params=leaf_params)
# graph_data.view(filename='xgboost_tree_customer_churn') ## save as PDF
