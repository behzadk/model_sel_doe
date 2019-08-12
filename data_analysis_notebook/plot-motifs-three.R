library(igraph)

args <- commandArgs(trailingOnly = TRUE)

adjacency_mat_dir <- args[1]
data_dir <- args[2]
output_dir <- args[3]

output_path <- paste(output_dir, "ordered_topos.pdf", sep="")
ordred_model_txt_path <- paste(data_dir, "model_order.txt", sep="")
print(output_path)
files <- list.files(path=adjacency_mat_dir, pattern="*.csv", full.names=TRUE, recursive=FALSE)

# Get model numbers
model_nums <- c()
for (f in files) {
   file_name <- basename(f)
   model_num <- strsplit(file_name, "_")[[1]][2]
   model_nums <- cbind(model_nums, strtoi(model_num))
}
model_nums <- sort(model_nums)
model_nums <- as.list(read.table(ordred_model_txt_path))[[1]]

# Get species names
a_table <- read.csv(files[1])
species_names <- names(a_table[, 2:ncol(a_table)])
species_names <- c(expression(N[1]), expression(N[2]),  expression(N[3]), expression(B[1]), expression(B[2]), expression(A[1]), expression(A[2]))

model_titles <- c("")

concat_df <- NULL

file_name_template <- "model_#NUM#_adj_mat.csv"
idx <- 1
# Iterate models in ascending order
for (m_num in model_nums) {
   f <- paste(adjacency_mat_dir, sub("#NUM#",toString(m_num), file_name_template), sep="")
   file_name <- basename(f)
   model_num <- strsplit(file_name, "_")[[1]][2]
   model_titles <- rbind(model_titles, model_num)

   x <- read.csv(f)[, 1:length(species_names)+1]
   flat_list <- c()
   for (i in 1:nrow(x)) {
      flat_list <- c(flat_list, unlist(x[i,], use.names=F))
   }
   concat_df <- rbind(concat_df, flat_list)
}
# print(concat_df)

d <- concat_df
# d <- read.table("Mushroom_structures_T1_edit.txt",header=F)
# names(d) <- c("yuu", "yvu", "ywu", "yuv", "yvv", "ywv", "yuw", "yvw", "yww")


if(1){
   # plot only core topologies
   # dd <- d[,-c(1,5,9)]
   dd <- d
   dd <- unique(dd)

   nedge <- apply(dd,MARGIN=1,function(x){ sum(abs(x)) } )
   # dd <- dd[order(nedge),]

   dd <- rbind( c(0,0,0,0,0,0,0), dd)

   # u, v, w: 0, 1, 2
   elist <- rbind( 
      c(1,1), c(2,1), c(3,1), c(4, 1), c(5, 1), c(6, 1), c(7, 1),
      c(1,2), c(2,2), c(3,2), c(4,2), c(5,2), c(6,2), c(7, 2),
      c(1,3), c(2,3), c(3,3), c(4,3) , c(5,3), c(6,3), c(7, 3),
      c(1,4), c(2,4), c(3,4), c(4,4) , c(5,4), c(6,4),c(7, 4),
      c(1,5), c(2,5), c(3,5), c(4,5) , c(5,5), c(6,5),c(7, 5),
      c(1,6), c(2,6), c(3,6), c(4,6) , c(5,6), c(6,6),c(7, 6),
      c(1,7), c(2,7), c(3,7), c(4,7) , c(5,7), c(6,7), c(7, 7)
      )


   layout =  5.3 * rbind( c(0,0), c(0,1), c(1,1), c(1,2), c(2,2), c(2,3)  )

   layout =  1 * rbind( c(-1.25, -1.25), c(1.25, 1.25), c(1.25, -1.25), c(0, -1.25), c(1.25,0), c(-1.25, 0), c(0, 1.25)  )

   #x11(height=10,width=18)
   pdf(output_path,height=8,width=22)
   par(mfrow=c(3,6))
   
   for(i in c(1:nrow(dd)) ){
      code <- dd[i,]
      edges <- elist[ which(abs(code) == 1), ]
      edges.col <- ifelse( code[ abs(code) == 1 ] == 1, "black", "red")

      l <- layout

      g <- graph_from_edgelist(edges, directed=TRUE )
      plot(g, vertex.color="white", edge.curved=0.3, vertex.label=species_names, edge.color=edges.col, layout=l, rescale=F, vertex.size = 50, edge.width=2, vertex.label.cex=2)
      title(main=model_titles[i], adj=0, cex.main=3)

   }

   dev.off()
}