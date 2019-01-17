library(ggplot2)
library(stringr)
theme_set(theme_classic())
library(sm)
library(gridExtra)
library(gtable)
options(error=traceback)


theme0 <- function(...) theme( legend.position = "none",
	                               panel.background = element_blank(),
	                               #panel.grid.major = element_blank(),
	                               panel.grid.minor = element_blank(),
	                               panel.margin = unit(0,"null"),
	                               axis.ticks = element_blank(),
	                               #axis.text.x = element_blank(),
	                               #axis.text.y = element_blank(),
	                               #axis.title.x = element_blank(),
	                               axis.title.y = element_blank(),
	                               axis.ticks.length = unit(0,"null"),
	                               axis.ticks.margin = unit(0,"null"),
	                               axis.line = element_blank(),
	                               panel.border=element_rect(color=NA),
	                               ...)

theme_left_dens <- function(...) theme( legend.position = "none",
                               panel.background = element_blank(),
                               #panel.grid.major = element_blank(),
                               panel.grid.minor = element_blank(),
                               panel.spacing = unit(1,"null"),
                               axis.ticks = element_blank(),
                               axis.text.x = element_blank(),
                               axis.text.y = element_blank(),
								axis.title.x = element_blank(),
                               axis.title.y = element_blank(),
                               axis.ticks.length = unit(0,"null"),
                               axis.ticks.margin = unit(0,"null"),
                               axis.line = element_blank(),
                               panel.border=element_rect(color=NA),
                               #plot.margin = unit(c(-0.25, 0.4, -0.25, 0.4),"lines"),
                               ...)

theme_top_dens <- function(...) theme( legend.position = "none",
                               	panel.background = element_blank(),
                               	#panel.grid.major = element_blank(),
                               	panel.grid.minor = element_blank(),
                               	panel.margin = unit(0,"null"),
                               	axis.ticks = element_blank(),
                               	axis.text.x = element_blank(),
                               	axis.text.y = element_blank(),
								axis.title.x = element_blank(),
                               	axis.title.y = element_blank(),
                               	axis.ticks.length = unit(0,"null"),
                               	axis.ticks.margin = unit(0,"null"),
                               	axis.line = element_blank(),
                               	panel.border=element_rect(color=NA),
                               	# plot.margin = unit(c(0, 0.4, 0.4, 0.4),"lines"),
                               ...)

theme_contour <- function(...) theme( legend.position = "none",
                               	panel.background = element_blank(),
                               	#panel.grid.major = element_blank(),
                               	panel.grid.minor = element_blank(),
                               	panel.spacing = unit(0,"null"),
                               	axis.ticks = element_blank(),
								axis.text.x = element_blank(),
                               	axis.text.y = element_blank(),
								axis.title.x = element_blank(),
                               	axis.title.y = element_blank(),
                               	axis.ticks.length = unit(0,"null"),
                               	axis.ticks.margin = unit(0,"null"),
                               	axis.line = element_blank(),
                               	#panel.border=element_rect(color=NA),
	                               ...)

theme_empty <- function(...) theme(plot.background = element_blank(), 
	                           panel.grid.major = element_blank(), 
	                           panel.grid.minor = element_blank(), 
	                           panel.border = element_blank(), 
	                           panel.background = element_blank(),
	                           axis.title.x = element_blank(),
	                           axis.title.y = element_blank(),
	                           axis.text.x = element_blank(),
	                           axis.text.y = element_blank(),
	                           axis.ticks = element_blank(),
	                           axis.line = element_blank(),
                               )

convert_to_grid <- function(x, y) {
	nrows = length(x)
	ncolmns = length(y)

	new_y = c()
	new_x = c()
	print(ncolmns)
	count = 1
	for (y_coor in y){
		for (x_coor in x){
			new_x[count] <- x_coor
			new_y[count] <- y_coor
			count = count + 1
		}
	}

	new_df <- data.frame(new_x, new_y)
	colnames(new_df) <- c('x', 'y')

	return(new_df)
}

make_contour_plot <-function(x_data, y_data, x_lims, y_lims, weights_data, true_val_x, true_val_y) {
	dens <- sm.density( cbind(x_data, y_data), weights=weights_data, display="none", nbins=0)
	x1 = dens$eval.points[,1]
	y1 = dens$eval.points[,2]
	z1 = dens$estimate

	# Generate coordinates corresponding to z grid
	dens_df <- convert_to_grid(x1, y1)
	colnames(dens_df) <- cbind("x1", "y1")
	dens_df$z1 <- c(z1)

	# Filling a column with same values so they can be passed in the same df
	dens_df$true_val_x <- c(rep(true_val_x, length(x1)))
	dens_df$true_val_y <- c(rep(true_val_y, length(y1)))



	pCont_geom <- ggplot(data=dens_df) + 
	geom_contour(aes(x=x1, y=y1, z=z1, colour="red"), bins=10) +

	##geom_point(aes(x=true_val_x, y=true_val_y, colour="blue")) +
	#geom_point(aes(x=true_val_x[1], y=true_val_y[1], colour="blue")) + 

	geom_hline(yintercept=true_val_y[1], linetype="dashed", color="blue") +
	geom_vline(xintercept=true_val_x[1], linetype="dashed", color="blue") +

	scale_x_continuous(name="x", limits = x_lims, expand = c(0,0)) + 
	scale_y_continuous(position="right", limits = y_lims, expand = c(0,0)) + 
	theme_bw() + 
	theme_contour()

	print(true_val_y)



	return(pCont_geom)
}

make_dual_contour_plot <-function(x1_data, y1_data, x2_data, y2_data, x_lims, y_lims, weights_data_1, weights_data_2) {
	dens_1 <- sm.density( cbind(x1_data, y1_data), weights=weights_data_1, display="none", nbins=0 )
	dens_2 <- sm.density( cbind(x2_data, y2_data), weights=weights_data_2, display="none", nbins=0 )

	# Extract data from sm.density functions
	x1 = dens_1$eval.points[,1]
	y1 = dens_1$eval.points[,2]
	z1 = dens_1$estimate

	x2 = dens_2$eval.points[,1]
	y2 = dens_2$eval.points[,2]
	z2 = dens_2$estimate

	# Generate grid coordinates and df for each parameter set
	dens_1_df <- convert_to_grid(x1, y1)
	colnames(dens_1_df) <- cbind("x1", "y1")

	dens_2_df <- convert_to_grid(x2, y2)
	colnames(dens_2_df) <- cbind("x2", "y2")

	# Add z data to each parameter set
	dens_1_df$z1 <- c(z1)
	dens_2_df$z2 <- c(z2)

	# Combine dataframes
	dens_df_combined <- cbind(dens_1_df, dens_2_df)

	pCont_geom <- ggplot(data=dens_df_combined) + 
	geom_contour(aes(x=x1, y=y1, z=z1, colour="red"), bins=10) + 
	geom_contour(aes(x=x2, y=y2, z=z2, colour="blue"), bins=10) + 
	scale_x_continuous(name="x", limits = x_lims, expand = c(0,0)) + 
	scale_y_continuous(position="right", limits=y_lims, expand = c(0,0)) + 
	theme_bw() + 
	theme_contour()

	return(pCont_geom)
}


make_annotation_plot <- function(annot_text) {
	# plot_str <- paste(annot_text)

	pAnnot <- ggplot() + 
  	annotate("text", x = 4, y = 25, size=4, label = paste(annot_text), parse=TRUE) + 
  	theme_bw() +
  	theme_empty()

	
	return(pAnnot)
}

make_top_plot <- function(x_data, x_lims, weights_data) {
	plot_df <- data.frame(x_data, weights_data)
	colnames(plot_df) <- c("x", "w")
	pTop <- ggplot(data=plot_df) +
  	geom_density(aes(x= x, weight=w, colour = 'red')) +
  	scale_x_continuous(name = 'log10(GFP)', limits=x_lims, expand = c(0,0)) + 
  	scale_y_continuous(position="right", expand = c(0,0)) + 
  	theme_bw() + theme_top_dens()
  	return(pTop)
}

make_dual_top_plot <- function(x_data_1, x_data_2,  x_lims, weights_data_1, weights_data_2) {
	plot_df <- data.frame(x_data_1, x_data_2, weights_data_1, weights_data_2)
	colnames(plot_df) <- c("x1", "x2", "w1", "w2")

	pTop <- ggplot(data=plot_df) +
  	geom_density(aes(x= x1, weight=w1, colour = 'red')) +
  	geom_density(aes(x= x2, weight=w2, colour = 'blue')) +
  	scale_x_continuous(name = 'log10(GFP)', limits=x_lims, expand = c(0,0)) + 
  	scale_y_continuous(position="right", expand = c(0,0)) + 
  	theme_bw() + theme_top_dens()
  	return(pTop)
}


make_left_plot <- function(x_data, x_lims, weights_data) {
	plot_df <- data.frame(x_data, weights_data)
	colnames(plot_df) <- c("x", "w")

	pLeft <- ggplot(data=plot_df) +
  	geom_density(aes(x= x, weight=w, colour = 'red')) +
  	scale_x_continuous(name = 'log10(GFP)', position="top", limits = x_lims, expand = c(0,0))  + 
  	coord_flip() + 
  	scale_y_reverse() + 
  	theme_bw() + theme_left_dens()

  	return(pLeft)
}

make_dual_left_plot <- function(x_data_1, x_data_2, x_lims, weights_data_1, weights_data_2) {
	plot_df <- data.frame(x_data_1, x_data_2, weights_data_1, weights_data_2)
	colnames(plot_df) <- c("x1", "x2", "w1", "w2")

	pLeft <- ggplot(data=plot_df) +
  	geom_density(aes(x= x1, weight=w1, colour = 'red')) +
  	geom_density(aes(x= x2, weight=w2, colour = 'blue')) +
  	scale_x_continuous(name = 'log10(GFP)', position="top", limits = x_lims, expand = c(0,0))  + 
  	coord_flip() + 
  	scale_y_reverse() + 
  	theme_bw() + theme_left_dens()

  	return(pLeft)
}

make_empty_plot <- function() {
	pEmpty <- ggplot() + geom_point(aes(1,1), colour = 'white') +  theme_empty()

  	return(pEmpty)
}

plot_dens_2d_one_pop <- function(param_data, weights_data, cut, param_limits, output_name, true_values_vector) {
	# Plots densities and contours for one population on a grid with each par
	# vs another

	nptot <- dim(param_data)[2]
	pars <- c(0:nptot)

	# remove the cut parameters
	pars <- pars[ !(pars %in% cut) ]
	true_values_vector <- true_values_vector[ !(true_values_vector %in% cut) ]

	nParams <- length(pars)
	nCols <-  length(pars)
	nRows <- length(pars)

	# Generate top plots
	top_plots <- list()
	left_plots <- list()

	# Initiate empty plot list
	plot_list <- list()
	plot_list_index <- 1

	row_idx <- 1

	for (row in pars) {
		col_idx <- 1
		for (col in pars) {

			# Set top left tile to empty
			if ((row_idx == 1) & (col_idx ==1)) {
				plot_list[[plot_list_index]] <- make_empty_plot()
			}

			# Set top row to top_dens plots
			else if (row_idx == 1) {
				plot_list[[plot_list_index]] <- make_top_plot(param_data[,col], param_limits[,col], weights_data)
			}

			else if (col_idx == 1) {
				# Col == 1 we plot a left density
				plot_list[[plot_list_index]] <- make_left_plot(param_data[,row], param_limits[,row], weights_data)
			}

			# Set middle row to param name
			else if (col_idx == row_idx) {
				par_string <- get_name_idx(param_idx = col)
				plot_list[[plot_list_index]] <- make_annotation_plot(par_string)
			}

			# Plot contours for all other grid spaces
			else {
				plot_list[[plot_list_index]] <- make_contour_plot(param_data[,col], param_data[,row], param_limits[,col], param_limits[,row], weights_data, true_values_vector[col], true_values_vector[row])
			}

			plot_list_index = plot_list_index + 1
			col_idx = col_idx + 1
		}

		row_idx = row_idx + 1
		col_idx <- 1
	}

	print("starting grid arrange")
	
	# Set size of grid widths and heights
	width_list <- as.list(rep(4, nCols))
	height_list <- as.list(rep(4, nCols))

	pMar <- grid.arrange(grobs=plot_list, ncol=nCols, nrow=nRows, widths = width_list, heights = height_list)

	#pMar <- do.call("grid.arrange", c(plot_list, ncol=nCols+1, nrow=nRows+1))

	ggsave(output_name, pMar)
}


plot_dens_2d_two_pop <- function(param_data_1, param_data_2, weights_data_1, weights_data_2, cut, param_limits, output_name) {
	# Plots densities and contours of two parameters
	# against each other

	if ( dim(param_data_1)[2] != dim(param_data_2)[2] ) {
		print("param_data_1 and param_data_2 are not the same dimensions")
		quit()
	}


	# Set total number of parameters
	nptot <- dim(param_data_1)[2]
	pars <- c(0:nptot)

	# remove the cut parameters
	pars <- pars[ !(pars %in% cut) ]

	nCols = length(pars)
	nRows = length(pars)

	# Generate top plots
	top_plots <- list()
	left_plots <- list()

	# Initiate empty plot list
	plot_list <- list()
	plot_list_index <- 1

	row_idx <- 1

	for (row in pars) {
		col_idx <- 1

		for (col in pars) {

			# Set top left tile to empty
			if ((row_idx == 1) & (col_idx ==1)) {
				plot_list[[plot_list_index]] <- make_empty_plot()
			}

			# Set top row to top_dens plots
			else if (row_idx == 1) {
				plot_list[[plot_list_index]] <- make_dual_top_plot(param_data_1[,col], param_data_2[,col], param_limits[,col], weights_data_1, weights_data_2)
			}
		
			else if (col_idx == 1) {
				# Plot a left density in column 1
				plot_list[[plot_list_index]] <- make_dual_left_plot(param_data_1[,row], param_data_2[,row], param_limits[,row], weights_data_1, weights_data_2)
			}

			# Set middle row to param name
			else if (col_idx == row_idx) {
				par_string <- get_name_idx(param_idx = col)
				print(par_string)
				plot_list[[plot_list_index]] <- make_annotation_plot(par_string)
			}

			# Plot contours
			else {				
				plot_list[[plot_list_index]] <- make_dual_contour_plot(param_data_1[,col], param_data_1[,row],
					param_data_2[,col], param_data_2[,row],
					param_limits[,col], param_limits[,row], 
					weights_data_1, weights_data_2)
			}

			plot_list_index = plot_list_index + 1
			col_idx = col_idx + 1
		}

		row_idx = row_idx + 1
		col_idx <- 1

	}


	print("starting grid arrange")
	print(length(plot_list))
	width_list <- as.list(rep(4, nCols))
	height_list <- as.list(rep(4, nCols))
	pMar <- grid.arrange(grobs=plot_list, ncol=nCols, nrow=nRows, widths = width_list, heights = height_list)

	ggsave(output_name, pMar)
}

get_name_idx1 <- function(string_ref=NULL, param_idx=NULL) {
	# If string ref supplied, name index is returned
	# If param index is supplied, expression of param is returned
    names = c("Ru","n1","ku","Rv","n2","kv","del","u0","v0","mu1","s1")

    expression_names_list <- list( expression(R[u]), expression(n[1]), expression(k[u]), expression(R[v]), expression(n[2]), expression(k[v]), expression(delta),
          expression(u[0]), expression(v[0]),
	  expression(mu[u]), expression(sigma[u]) )

    if(!is.null(string_ref)) {
	    for (i in seq_along(names)) {
	    	if (string_ref == names[[i]]) {
	    		return(i)
	    	}
    	}
    }

    if (!is.null(param_idx)) {
    	print(names[[param_idx]])
    	return(expression_names_list[[param_idx]])
    }
}

get_name_idx2 <- function(string_ref=NULL, param_idx=NULL) {
	# If string ref supplied, name index is returned
	# If param index is supplied, expression of param is returned
    names = c("Ru","n1","ku","Rv","n2","kv","del","u0","v0","mu1","s1","mu2","s2")

    expression_names_list <- list( expression(R[u]), expression(n[1]), expression(k[u]), expression(R[v]), expression(n[2]), expression(k[v]), expression(delta),
          expression(u[0]), expression(v[0]),
	  expression(mu[u]), expression(sigma[u]),
	  expression(mu[v]), expression(sigma[v]) )

    if(!is.null(string_ref)) {
	    for (i in seq_along(names)) {
	    	if (string_ref == names[[i]]) {
	    		return(i)
	    	}
    	}
    }

    if (!is.null(param_idx)) {
    	print(names[[param_idx]])
    	return(expression_names_list[[param_idx]])
    }
}

############################################

w1 <- read.table("run3/res-sim-1D-1/pop8/data-weights.txt",header=F)[[1]]
d1 <- read.table("run3/res-sim-1D-1/pop8/data-posteriors-dyn.txt",header=F)
i1 <- read.table("run3/res-sim-1D-1/pop8/data-posteriors-init.txt",header=F)
m1 <- read.table("run3/res-sim-1D-1/pop8/data-posteriors-mu.txt",header=F)
s1 <- read.table("run3/res-sim-1D-1/pop8/data-posteriors-sg.txt",header=F)


M1<- cbind(d1,i1,m1,s1)
names(M1) <- c("Ru","n1","ku","Rv","n2","kv","del","u0","v0","mu1","s1")

lims <- cbind( c(100,200),c(1,4),c(50,150),c(100,200),c(1,4),c(50,150),c(0,10),
               c(0,20), c(100,200),
	       c(0,2), c(4,6) )

to_cut <- c(0)

true_vals <- c(160,2,100,160,2,100,5,1,160,1,5)

output_name <-  "sims-posterior-M1-tv.pdf"
get_name_idx <- get_name_idx1
plot_dens_2d_one_pop(M1, w1, to_cut, lims, output_name, true_vals)

############################################

w2 <- read.table("run3/res-sim-2D-1/pop8/data-weights.txt",header=F)[[1]]
d2 <- read.table("run3/res-sim-2D-1/pop8/data-posteriors-dyn.txt",header=F)
i2 <- read.table("run3/res-sim-2D-1/pop8/data-posteriors-init.txt",header=F)
m2 <- read.table("run3/res-sim-2D-1/pop8/data-posteriors-mu.txt",header=F)
s2 <- read.table("run3/res-sim-2D-1/pop8/data-posteriors-sg.txt",header=F)


M2 <- cbind(d2,i2,m2[[1]],s2[[1]],m2[[2]],s2[[2]])
names(M2) <- c("Ru","n1","ku","Rv","n2","kv","del","u0","v0","mu1","s1","mu2","s2")

lims <- cbind( c(100,200),c(1,4),c(50,150),c(100,200),c(1,4),c(50,150),c(0,10),
               c(0,20), c(100,200),
	       c(0,2), c(4,6), c(0,2), c(4,6) )

to_cut <- c(0)

true_vals <- c(160,2,100,160,2,100,5,1,160,1,5,1,5)

output_name <-  "sims-posterior-M2-tv.pdf"
get_name_idx <- get_name_idx2
plot_dens_2d_one_pop(M2, w2, to_cut, lims, output_name, true_vals)