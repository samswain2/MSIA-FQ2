\ Model Beer_and_Ale
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  13 ale + 23 beer
Subject To
 corn: 5 ale + 15 beer <= 480
 hops: 4 ale + 4 beer <= 160
 malt: 35 ale + 20 beer <= 1190
Bounds
End
