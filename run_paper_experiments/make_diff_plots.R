require(dplyr)
require(ggplot2)

mm_error_diff_plot <- function(data, unit, measure, error, 
                          title, subtitle, note=NULL,
                          ylim_ub=NA) {

  # https://stackoverflow.com/questions/42100892/how-to-pass-a-string-to-dplyr-filter-in-a-function
  
  plt <- data %>% 
    filter(unit == UQ(unit),
           qdata != 'onlyq', 
           model %in% c('lasso', 'lm', 'poly', 'nnet', 'rf', 'svr'),
           measure == UQ(measure)) %>% 
    ggplot()
  
    if (error == 'mape') {
      plt <- plt + geom_point(aes(x=model, y=test_mape_diff, color=qdata, shape=qdata), 
                 size=4) + labs(y = "Change in MAPE")
    }  else {
      plt <- plt + geom_point(aes(x=model, y=test_mae_diff, color=qdata, shape=qdata), 
                              size=4) + labs(y = "Change in MAE")
    }
    
  plt <- plt +
    labs(
      title = title,
      subtitle = subtitle,
      caption = note,
      tag = NULL,
      x = "Model type",
      colour = "Feature set",
    ) + 
    scale_colour_discrete(name  ="Queueing features",
                          breaks=c("noq", "basicq", "q"),
                          labels=c("None", "Basic", "Advanced")) +
    scale_shape_discrete(name  ="Queueing features",
                         breaks=c("noq", "basicq", "q"),
                         labels=c("None", "Basic", "Advanced")) + 
    theme_bw() +
    theme(plot.title = element_text(hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5)) +
    theme(
      legend.text = element_text(size = 8),
      legend.title = element_text(size = 9),
    ) + guides(shape = guide_legend(override.aes = list(size = 2)))
  
  return(plt)
  
}

