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

get_name_idx <- function(string_ref=NULL, param_idx=NULL) {
	# If string ref supplied, name index is returned
	# If param index is supplied, expression of param is returned

    names = list(
    	"D", "mu_max_1", "mu_max_2", "KB_1", "KB_2",
    	"kA_1", "kA_2", "K_omega_1", "K_omega_2", "kBmax_1",
    	"kBmax_2", "nB_1", "nB_2", "n_omega_1", "n_omega_2",
    	"omega_max_1", "omega_max_2", "N_1", "N_2"
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





		# expression(Kc), expression(omega_c_max),
	 #    expression(K[omega]), expression(n[omega]), expression(S[0]), expression(gX), expression(gC),
	 #    expression(C0L), expression(KDL),    
	 #    expression(nL), expression(K1L), expression(K2L), expression(gamma[maxL]),
	 #    expression(K1T), expression(K2T), expression(gamma[maxT]), expression(C0B), 
	 #    expression(LB), expression(NB),expression(KDB) ,expression(K1B), 
	 #    expression(K2B), expression(K3B), expression(gamma[maxB]), expression(cgt), expression(k_alpha_max),
	 #    expression(k_beta_max),
	 #    expression(X), expression(C), expression(S), expression(B),
	 #    expression(A))


    if(!is.null(string_ref)) {
	    for (i in seq_along(names)) {
	    	if (string_ref == names[[i]]) {
	    		return(expression_names_list[[i]])
	    	}
    	}
    }

    if (!is.null(param_idx)) {
    	print(names[[param_idx]])
    	return(expression_names_list[[param_idx]])
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
	# geom_point(aes(x=true_val_x, y=true_val_y, colour="blue")) + 
	scale_x_continuous(name="x", limits = x_lims, expand = c(0,0)) + 
	scale_y_continuous(position="right", limits=y_lims, expand = c(0,0)) + 
	theme_bw() + 
	theme_contour()




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
  	annotate("text", x = 4, y = 25, size=8, label = paste(annot_text), parse=TRUE) + 
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
	print(pars)

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
				print(names(param_data)[col])
				par_string <- get_name_idx(string_ref = names(param_data)[col])
				print(par_string)
				plot_list[[plot_list_index]] <- make_annotation_plot(par_string)
			}

			# Plot contours for all other grid spaces
			else {
				plot_list[[plot_list_index]] <- make_contour_plot(param_data[,col], param_data[,row], param_limits[,col], param_limits[,row], weights_data, true_values_vector[col_idx -1], true_values_vector[row_idx - 1])
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


get_fixed_parameter_columns <- function(data_df) {

	fixed_param_list = c()

	idx  = 1
	for(i in names(data_df)){
		x = length(unique(data_df[, i]))
		print(x)
		if (x <= 1) {
			fixed_param_list <- c(fixed_param_list, idx)
		}

		idx = idx + 1
	}
	return(fixed_param_list)
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

wd <- "/home/behzad/Documents/barnes_lab/cplusplus_software/speed_test/repressilator/cpp/"
data_dir <- paste(wd, "output/two_species_stable_12/Population_0/model_sim_params/", sep="")
data_path <- paste(data_dir, "model_119_all_params", sep="")
data_df <- read.csv(data_path)


param_lims <- make_param_lims(data_df[, 4:ncol(data_df)])
df_list <- split(data_df, f= data_df$Accepted)
rejected_df <- df_list[[1]]
accepted_df <- df_list[[2]]
dim(rejected_df)
accepted_df <- accepted_df[, 4:ncol(accepted_df)]
rejected_df <- rejected_df[, 4:ncol(rejected_df)]

fixed_params = get_fixed_parameter_columns(accepted_df)
to_cut <- fixed_params

keep_columns <- c("D", "mu_max_1", "mu_max_2", "omega_max_1", "omega_max_2", "N_1", "N_2")
remove_columns <- setdiff(names(accepted_df), keep_columns)
print(remove_columns)
length(remove_columns)
length(keep_columns)
length(names(accepted_df))

to_cut <- c()
idx <- 1
for (name in names(accepted_df)) {
	if (name %in% remove_columns) {
		to_cut <- cbind(to_cut, idx)
	}
	idx <- idx + 1
}

print("to cut:")
print(to_cut)
print("")

weights <- rep(1, dim(accepted_df)[1])
output_name <-  "model_119_pop_2D_dens.pdf"
dummy_true_val_vector <- rep(0.8,  dim(accepted_df)[2])

# dim(accepted_df)
# dim(param_lims)
plot_dens_2d_one_pop(accepted_df, weights, to_cut, param_lims, output_name, dummy_true_val_vector)

quit()
