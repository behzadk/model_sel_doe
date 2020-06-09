library(ggplot2)
library(stringr)
theme_set(theme_classic())
library(sm)
library(gridExtra)
library(gtable)
options(error=traceback)
library(ggisoband)
library(viridis)
library(scales)

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
                               # panel.grid.major = element_blank(),
                               # panel.grid.minor = element_blank(),
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
                               # plot.margin = unit(c(-0.25, 0.4, -0.25, 0.4),"lines"),
                               ...)

theme_top_dens <- function(...) theme( legend.position = "none",
                               	panel.background = element_blank(),
                               	# panel.grid.major = element_blank(),
                               	# panel.grid.minor = element_blank(),
                               	panel.margin = unit(0,"null"),
                               	# axis.ticks = element_blank(),
                               	axis.text.x = element_blank(),
                               	axis.text.y = element_blank(),
								axis.title.x = element_blank(),
                               	axis.title.y = element_blank(),
                               	axis.ticks.length = unit(0,"null"),
                               	axis.ticks.margin = unit(0,"null"),
                               	# axis.line = element_blank(),
                               	panel.border=element_rect(color=NA),
                               	# plot.margin = unit(c(0, 0.4, 0.4, 0.4),"lines"),
						    	legend.background = element_rect(fill = "transparent"), 
    							legend.box.background = element_rect(fill = "transparent"), # get rid of legend panel bg
                               ...)

theme_1D_plot <- function(...) theme( legend.position = "none",
                               	panel.background = element_blank(),
                               	panel.grid.major = element_blank(),
                               	panel.grid.minor = element_blank(),
                               	panel.margin = unit(0,"null"),
                               	axis.ticks = element_blank(),
                               	axis.text.x = element_blank(),
                               	axis.text.y = element_blank(),
								# axis.title.x = element_blank(),
                               	axis.title.y = element_blank(),
                               	axis.ticks.length = unit(0,"null"),
                               	axis.ticks.margin = unit(0,"null"),
                               	axis.line = element_blank(),
                               	panel.border=element_rect(color=NA),
                               	plot.margin = unit(c(0.5, 1.2, 0.5, 1.2),"lines"),
                               ...)



theme_contour <- function(...) theme( legend.position = "none",
                               	panel.background = element_rect(fill = "transparent",colour = NA),
                               	panel.grid.major = element_blank(),
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
								plot.background = element_rect(fill = "transparent", color = NA),
						    	legend.background = element_rect(fill = "transparent"), 
    							legend.box.background = element_rect(fill = "transparent"), # get rid of legend panel bg
						    	# get rid of legend bg
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

fancy_scientific <- function(l) {
     # turn in to character string in scientific notation
     l <- format(l, scientific = TRUE)
     # quote the part before the exponent to keep all the digits
     l <- gsub("^(.*)e", "'\\1'e", l)
     # turn the 'e+' into plotmath format
     l <- gsub("e", "%*%10^", l)
     # return this as an expression
     parse(text=l)
}

get_name_idx <- function(string_ref=NULL, param_idx=NULL) {
	# If string ref supplied, name index is returned
	# If param index is supplied, expression of param is returned

    names = list(
		"D", "g", "g_trp", "K_A_B", "K_A_I", "K_A_T", "K_A_V", 
		"k_I", "K_mu", "K_mu_trp", "k_omega_B", "k_omega_T", "k_TV_ann", 
		"kA_1", "kB_max", "kI_max", "kT_max", "kV_max", "mu_max_1", "mu_max_2", "nI", "n_A_B", 
		"n_A_I", "n_A_T", "n_A_V", "p", "S0", "S0_trp", "omega_max_1", "n_omega_B", "n_omega_T",
		"N_1", "N_2", "S_glu", "S_trp", "B", "A", "V", "I", "T"
		)



		# "kA_1", "K_omega", "n_omega", "S0", "gX", 
    	# "gC", "C0L", "KDL","nL","K1L",
    	# "K2L", "ymaxL", "K1T", "K2T", "ymaxT", 
    	# "C0B", "LB", "NB", "KDB", "K1B", 
    	# "K2B", "K3B", "ymaxB", "cgt", "k_alpha_max", 
    	# "k_beta_max", "X", "C", "S", 
    	# "B", "A"
	expression_names_list <- list(
		expression(D), expression(mu[max[1]]), expression(mu[max[2]]), expression(K[B[1]]), expression(K[B[mccV]]),
		expression(kA[1]), expression(kA[2]), expression(K[omega[1]]),  expression(K[omega[2]]), expression(kBmax[1]), 
		expression(kBmax[2]), expression(nB[1]), expression(nB[2]), expression(n[omega[1]]), expression(n[omega[2]]),
		expression(omega[max[1]]), expression(omega[max[2]]), expression(N[1]), expression(N[2]) 
		)


    if(!is.null(string_ref)) {
	    for (i in seq_along(names)) {
	    	if (string_ref == names[[i]]) {
	    		# print(names[[i]])
	    		return(names[[i]])
	    		# return(expression_names_list[[i]])
	    	}
    	}
    }

    if (!is.null(param_idx)) {
    	return(names[[param_idx]])

    	# return(expression_names_list[[param_idx]])
    }
}


convert_to_grid <- function(x, y) {
	nrows = length(x)
	ncolmns = length(y)

	new_y = c()
	new_x = c()
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


make_dual_contour_plot <-function(x1_data, y1_data, x2_data, y2_data, x_lims, y_lims, weights_data_1, weights_data_2) {
	y_trans_scale <- "identity"
	x_trans_scale <- "identity"

    if ((x_lims[1] < 1e-4 && x_lims[1] != 0) || (x_lims[1] > 1e4)){
    	x_trans_scale <- "log10"
    }

    if ((y_lims[1] < 1e-4 && y_lims[1] != 0) || (y_lims[1] > 1e4)){
    	y_trans_scale <- "log10"
    }

	if (identical("log10", x_trans_scale)) {
		x_lims <- log(x_lims)
	}

	if (identical("log10", y_trans_scale)) {
		y_lims <- log(y_lims)
	}

	print(x_trans_scale)


	dens_1 <- sm.density( cbind(x1_data, y1_data), weights=weights_data_1, display="none", nbins=0 )
	dens_2 <- sm.density( cbind(x2_data, y2_data), weights=weights_data_2, display="none", nbins=0 )

	# Extract data from sm.density functions
	x1 = dens_1$eval.points[,1]
	y1 = dens_1$eval.points[,2]

	x2 = dens_2$eval.points[,1]
	y2 = dens_2$eval.points[,2]


	# Generate grid coordinates and df for each parameter set
	dens_1_df <- convert_to_grid(x1, y1)

	dens_2_df <- convert_to_grid(x2, y2)

	# Add z data to each parameter set
	dens_1_df$z1 <- c(dens_1$estimate)
	dens_2_df$z2 <- c(dens_2$estimate)

	colnames(dens_1_df) <- cbind("x1", "y1", "z1")
	colnames(dens_2_df) <- cbind("x2", "y2", "z2")

	# Combine dataframes
	dens_df_combined <- cbind(dens_1_df, dens_2_df)

	pCont_geom <- ggplot(data=dens_df_combined) + 
	geom_contour(aes(x=x1, y=y1, z=z1, colour="blue"), bins=10) + 
	geom_contour(aes(x=x2, y=y2, z=z2, colour='#058989'), bins=10) + 
	scale_x_continuous(name="x", limits = x_lims, expand = c(0,0)) + 
	scale_y_continuous(position="right", limits=y_lims, expand = c(0,0)) + 
	theme_bw() + 
	theme_contour()

	return(pCont_geom)
}


make_annotation_plot <- function(annot_text) {
	# plot_str <- paste(annot_text)
	pAnnot <- ggplot() + 
  	annotate("text", x = 4, y = 25, size=1.5, label = paste(annot_text), parse=TRUE) + 
  	theme_bw() +
  	theme_empty()

	return(pAnnot)
}


make_dual_top_plot <- function(x_data_1, x_data_2,  x_lims, weights_data_1, weights_data_2) {
	x_1_df = data.frame(x_data_1, weights_data_1)
	x_2_df = data.frame(x_data_2, weights_data_2)

	colnames(x_1_df) <- c("x", "w")
	colnames(x_2_df) <- c("x", "w")

    if ((x_lims[1] < 1e-4 && x_lims[1] != 0) || (x_lims[1] > 1e4)){
    	x_lims <- log(x_lims)
    }

	# plot_df <- data.frame(x_data_1, x_data_2, weights_data_1, weights_data_2)
	# colnames(plot_df) <- c("x1", "x2", "w1", "w2")

	pTop <- ggplot() +
  	geom_density(aes(x= x_1_df$x, weight=x_1_df$w, y=..scaled.., colour = 'blue')) +
  	geom_density(aes(x= x_2_df$x, weight=x_2_df$w, y=..scaled.., colour = '#058989')) +
  	scale_x_continuous(name = 'log10(GFP)', limits=x_lims, expand = c(0,0)) +
  	scale_y_continuous(position="right", expand = c(0,0)) + 
  	theme_bw() + theme_top_dens()
  	return(pTop)
}



make_dual_left_plot <- function(x_data_1, x_data_2, x_lims, weights_data_1, weights_data_2) {
	x_1_df = data.frame(x_data_1, weights_data_1)
	x_2_df = data.frame(x_data_2, weights_data_2)
	colnames(x_1_df) <- c("x", "w")
	colnames(x_2_df) <- c("x", "w")

    if ((x_lims[1] < 1e-4 && x_lims[1] != 0) || (x_lims[1] > 1e4)){
    	x_lims <- log(x_lims)
    }

	pLeft <- ggplot() +
  	geom_density(aes(x= x_1_df$x, weight=x_1_df$w, y=..scaled.., colour = 'blue')) +
  	geom_density(aes(x= x_2_df$x, weight=x_2_df$w, y=..scaled.., colour = '#058989')) +
  	scale_x_continuous(name = 'log10(GFP)', limits=x_lims, expand = c(0,0)) +
  	scale_y_continuous(position="right", expand = c(0,0)) + 
	coord_flip() + 
  	scale_y_reverse() + 
  	theme_bw() + theme_top_dens()
  	return(pLeft)
}

make_empty_plot <- function() {
	pEmpty <- ggplot() + geom_point(aes(1,1), colour = 'white') +  theme_empty()

  	return(pEmpty)
}


plot_dens_2d_two_pop <- function(param_data_1, param_data_2, weights_data_1, weights_data_2, cut, param_limits, output_name) {
	# Plots densities and contours of two parameters
	# against each other

	if ( dim(param_data_1)[2] != dim(param_data_2)[2] ) {
		print("bad params")
		print(param_data_1)
		print("param_data_1 and param_data_2 are not the same dimensions")
		quit()
	}

	# Set total number of parameters
	nptot <- dim(param_data_1)[2]
	pars <- c(0:nptot)

	# remove the cut parameters
	pars <- pars[ !(pars %in% cut) ]
	# quit()

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
				# plot_list[[plot_list_index]] <- make_empty_plot()
				par_string <- names(param_data_1)[col]
				# par_string <- get_name_idx(param_idx = col)
				plot_list[[plot_list_index]] <- make_annotation_plot(par_string)
			}

			# Plot contours
			else {
				# plot_list[[plot_list_index]] <- make_empty_plot()

				plot_list[[plot_list_index]] <- make_dual_contour_plot(param_data_1[,col], param_data_1[,row], 
					param_data_2[,col], param_data_2[,row], param_limits[, col], param_lims[, row], weights_data_1, weights_data_2)

				# 	param_data_2[,col], param_data_2[,row],
				# 	param_limits[,col], param_limits[,row], 
				# 	weights_data_1, weights_data_2)
			}

			plot_list_index = plot_list_index + 1
			print(plot_list_index)
			col_idx = col_idx + 1
		}

		row_idx = row_idx + 1
		col_idx <- 1

	}


	# Set size of grid widths and heights
	width_list <- as.list(rep(4, nCols))
	height_list <- as.list(rep(4, nCols))

	pMar <- grid.arrange(grobs=plot_list, ncol=nCols, nrow=nRows, widths = width_list, heights = height_list)

	ggsave(output_name, pMar, bg="transparent")
}


plot_1d_one_pop <- function(param_data, weights_data, cut, param_limits, output_name, true_values_vector) 
{
	param_names <- names(accepted_df)
	# Plots densities and contours for one population on a grid with each par
	# vs another
	nptot <- dim(param_data)[2]
	pars <- c(1:nptot)

	# remove the cut parameters
	pars <- pars[ !(pars %in% cut) ]
	nParams <- length(pars)
	nCols <-  length(pars)
	nRows <- length(pars)

	# Initiate empty plot list
	plot_list <- list()
	plot_list_index <- 1
	for (p in pars) {
		plot_list[[plot_list_index]] <- make_1d_param_plot(param_data[,p], param_limits[,p], weights_data, param_names[[p]])
		plot_list_index <- plot_list_index + 1
	}

	# Set size of grid widths and heights
	width_list <- as.list(rep(3, nCols))
	height_list <- as.list(rep(3, nCols))

	pMar <- grid.arrange(grobs=plot_list, ncol=4)
	ggsave(output_name, pMar)
}

get_fixed_parameter_columns <- function(param_lims) {
	
	fixed_param_list = c()

	for(i in seq(from=1, to=dim(param_lims)[2], by=1)) {
		diff = param_lims[2, i] - param_lims[1, i]

		if (diff == 0 ) {
			fixed_param_list <- c(fixed_param_list, i)
		}
	}

	return(fixed_param_list)
}

make_param_lims_from_input <- function(output_params_df, input_params_file_path, input_species_file_path) {
	param_lims_list <- c()

	input_params_df <- read.table(input_params_file_path, sep=",")

	# Iterate parameter names
	for(i in names(output_params_df)){
		idx  = 1
		for(j in input_params_df[[1]]) {

			if(i == j) {

				min_x = (input_params_df[[2]][idx])
				max_x = (input_params_df[[3]][idx])
				param_lims_list <- cbind(param_lims_list, c(min_x, max_x))
			}

			idx = idx + 1

		}
	}

	input_species_df <- read.table(input_species_file_path, sep=",")

	# Iterate parameter names
	for(i in names(output_params_df)){
		idx  = 1

		for(j in input_species_df[[1]]) {
			if(i == j) {

				min_x = (input_species_df[[2]][idx])
				max_x = (input_species_df[[3]][idx])
				param_lims_list <- cbind(param_lims_list, c(min_x, max_x))
			}

			idx = idx + 1

		}
	}

	return(param_lims_list)
}




make_param_lims <- function(params_data_df) {
	param_lims_list <- c()

	idx  = 1
	for(i in names(params_data_df)){
		min_x = min(data_df[i])
		max_x = max(data_df[i])

		param_lims_list <- cbind(param_lims_list, c(min_x, max_x))

		idx = idx + 1
	}

	return(param_lims_list)

}

args <- commandArgs(trailingOnly = TRUE)
params_1_path <- args[1]
params_2_path <- args[2]
param_priors_inputs_path <- args[3]
species_inputs_path <- args[4]
model_idx <- args[5]
output_name <- args[6]

params_1_df <- read.csv(params_1_path)
params_2_df <- read.csv(params_2_path)

drop_cols <- c('sim_idx', 'population_num', 'pop_balance', 'X')
params_1_df <- params_1_df[ , !(names(params_1_df) %in% drop_cols)]
params_2_df <- params_2_df[ , !(names(params_2_df) %in% drop_cols)]


weights_1 <- params_1_df$particle_weight

if(all(is.na(weights_1))) {
	weights_1 <- rep(1, length(weights_1))
}

weights_2 <- params_2_df$particle_weight
if(all(is.na(weights_2))) {
	weights_2 <- rep(1, length(weights_2))
}

drop_cols <- c('sim_idx', 'population_num', 'pop_balance', 'X', 'particle_weight', 'd3', 'd6', 'd9', 'exp_num', 'batch_idx', 'model_ref')

params_1_df <- params_1_df[ , !(names(params_1_df) %in% drop_cols)]
params_2_df <- params_2_df[ , !(names(params_2_df) %in% drop_cols)]


param_lims <- make_param_lims_from_input(params_1_df, param_priors_inputs_path, species_inputs_path)
# quit()
fixed_params = get_fixed_parameter_columns(param_lims)

to_cut <- fixed_params
# to_cut <- c()
keep_columns <- c("D", "kB_max_1", "kB_max_2", "kB_max_3")
# keep_columns <- c()

remove_columns <- setdiff(names(params_1_df), keep_columns)
# remove_columns <- c()

idx <- 1
for (name in names(params_1_df)) {
	if (name %in% remove_columns) {
		to_cut <- cbind(to_cut, idx)
	}
	idx <- idx + 1
}

# print(to_cut)
# print(length(param_lims))
# print(length(params_1_df))
# quit()

plot_dens_2d_two_pop(params_1_df, params_2_df, weights_1, weights_2, to_cut, param_lims, output_name)
