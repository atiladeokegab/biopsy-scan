# Export only the geometry and phenotype columns from the verified SingleCellExperiment.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L)
sce <- readRDS(args[[1L]])
cols <- attr(attr(sce, "colData", exact = TRUE), "listData", exact = TRUE)
fields <- c("patient_cohort", "patient_id", "image_name", "cell_number",
            "image_width", "image_height", "cell_x", "cell_y", "cell_metacluster")
stopifnot(is.list(cols), all(c(fields, "patient_disease_status") %in% names(cols)))
keep <- which(as.character(cols$patient_cohort) == "Basel" &
              as.character(cols$patient_disease_status) == "tumor")
output <- as.data.frame(lapply(cols[fields], function(x) x[keep]), stringsAsFactors = FALSE)
write.csv(output, args[[2L]], row.names = FALSE, na = "")
