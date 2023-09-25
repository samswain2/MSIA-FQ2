\ Model Cutting_Stock
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  5 orders_1 + 7 orders_2 + 9 orders_3
Subject To
 roll: 5 num_5 + 7 num_7 + 9 num_9 <= 20
 width_5: orders_1 >= 150
 width_7: orders_2 >= 200
 width_9: orders_3 >= 300
Bounds
Generals
 orders_1 orders_2 orders_3 num_5 num_7 num_9
End
