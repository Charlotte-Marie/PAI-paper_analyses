# Functions for preparation for machine learning

# Recode binary variables to 0.5 and -0.5, except for group-column ()
convert_bin_vars <- function(dataframe, group_variable) {
  is_binary <- sapply(dataframe, function(col) length(unique(col)) <= 2)
  for (col in names(dataframe)[is_binary]) {
    if (col != group_variable){
      dataframe[[col]] <- ifelse(dataframe[[col]] == 0, -0.5, 0.5)
    }
  }
  return(dataframe)
}

# Recode group variable to 0 and 1
recode_group <- function(df, group_variable, group_0, group_1){
  # group_0: name of group that should be recoded to 0
  # group_1: name of group that should be recoded to 1
  df[[group_variable]]<- as.factor(df[[group_variable]])
  df[[group_variable]] <- dplyr::recode(df[[group_variable]],group_0 = "0", group_1 = "1")
  return(df)
}

# Split a dataset into features.txt, labels.txt, and groups.txt
# Arguments: group_variable, outcome_variable!
split_and_save <- function(df, group_variable, outcome_variable, output_path){
  # df: dataframe
  # group_variable
  # outcome_variable
  # output_path
  exclude_columns_from_features <- c(group_variable, outcome_variable)
  features <- df[, setdiff(names(df), exclude_columns_from_features)]
  group <- df[,c(group_variable)]
  outcome <- df[,c(outcome_variable)]
  write.table(outcome, file  = file.path(output_path,"labels.txt"), sep = "\t",row.names = FALSE, col.names = TRUE)
  write.table(features, file  =  file.path(output_path,"features.txt"), sep = "\t",row.names = FALSE, col.names = TRUE)
  write.table(group, file  =  file.path(output_path,"groups.txt"), sep = "\t",row.names = FALSE, col.names = TRUE)
}