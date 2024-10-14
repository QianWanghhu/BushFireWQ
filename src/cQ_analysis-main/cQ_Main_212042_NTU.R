
# Main script to separate baseflow, delineate storm events, 
# and run c-Q hysteresis index analyses

# Clear memory
rm(list=ls())

#################
# LOAD PACKAGES #
#################

library(tidyverse)
library(viridis)

###################
# SET DIRECTORIES #
###################

# Define the input directory
input_dir <- "C:/Users/u1066632/OneDrive - Australian National University/WORKANU/Projects/BushFireWQ/src/cQ_analysis-main/"
output_dir <- "C:/Users/u1066632/OneDrive - Australian National University/WORKANU/Projects/BushFireWQ/output/CQ_analysis/"

#####################
# READ IN FUNCTIONS #
#####################

source(file.path(input_dir,"cQ_functions.R"))

################
# READ IN DATA #
################

# Read in 30-min discharge data
allInputData30Min <- read.csv(file.path(input_dir,"212042_Hourly.csv"))
# Specify constituent in data set name
dataSetName <- "212042_NTU"

# Chose constitution for plot axes labels (NO3, TOC, or turbidity)
constit <- "turbidity"

allInputData30Min$datetime <- as.POSIXct(allInputData30Min$datetime,format("%d/%m/%Y %H:%M"),tz="EST")
# Trim the dataset
# Start and end date are used to select data for a given time period.
select_year <- TRUE
start_date <- as.POSIXct('2020-1-01 00:00:00')
end_date <- as.POSIXct('2020-12-31 00:00:00')
year_sel <- "2020"
allInputData30Min <- allInputData30Min %>%
  filter(datetime >= start_date & datetime < end_date)

allInputData30Min <- allInputData30Min %>% 
  mutate(rescaled_conc = ((conc-min(conc))/(max(conc)-min(conc))*max(q_cms)))


# Vector containing candidate baseflow separation filter values
candidateFilterPara <- c(0.98,0.99)
# candidateFilterPara <- c(0.99)


# Vector containing candidate stormflow threshold values
candidateSfThresh <- c(0.1, 0.2, 0.5)
# candidateSfThresh <- c(0.2)


# Vector with interpolation intervals used for calculating HI
interp <- seq(0,1,0.01)

##########################################
# RUN ANALYSIS TO GET HYSTERESIS INDICES #
##########################################

batchRun1 <- batchRunBfAndEvSepForCQ(qInputs = allInputData30Min,
                                     bfSepPasses = 3,
                                     filterParam = candidateFilterPara,
                                     sfSmoothPasses = 4,
                                     sfThresh = candidateSfThresh,
                                     cInputs = allInputData30Min,
                                     timeStep = 60,
                                     minDuration = 12,
                                     maxDuration = 200)

eventsDataAll1 <- getAllStormEvents(batchRun = batchRun1,
                                    timestep_min = 60)

batchRunFlowsLF1 <- batchRunflowCompare(qData = allInputData30Min,
                                         bfSepPasses = 4,
                                         filterPara = candidateFilterPara,
                                         sfSmoothPasses = 4)

eventsData1 <- stormEventCalcs(batchRun = batchRun1,
                               timestep_min = 60)

stormCounts1 <- stormCounts(batchRun1)

hysteresisData1 <- getHysteresisIndices(batchRun = batchRun1,
                                        xForInterp = interp,
                                        eventsData = eventsData1)

######################
# EXPORT OUTPUT DATA #
######################
if(select_year) {
  write.csv(eventsData1,file = file.path(output_dir,paste(dataSetName,"_Year_", year_sel, "_StormEventSummaryData.csv",sep="")))
  write.csv(batchRunFlowsLF1,file = file.path(output_dir,paste(dataSetName,"_Year_", year_sel, "_DischargeData.csv",sep="")))
  write.csv(hysteresisData1,file = file.path(output_dir,paste(dataSetName,"_Year_", year_sel, "_HysteresisData.csv",sep="")))
  write.csv(eventsDataAll1,file = file.path(output_dir,paste(dataSetName,"_Year_", year_sel, "_AllCQData.csv",sep="")))
  write.csv(stormCounts1,file = file.path(output_dir,paste(dataSetName,"_Year_", year_sel, "_StormCounts.csv",sep="")))
} else {
  write.csv(eventsData1,file = file.path(output_dir,paste(dataSetName,"_StormEventSummaryData.csv",sep="")))
  write.csv(batchRunFlowsLF1,file = file.path(output_dir,paste(dataSetName,"_DischargeData.csv",sep="")))
  write.csv(hysteresisData1,file = file.path(output_dir,paste(dataSetName,"_HysteresisData.csv",sep="")))
  write.csv(eventsDataAll1,file = file.path(output_dir,paste(dataSetName,"_AllCQData.csv",sep="")))
  write.csv(stormCounts1,file = file.path(output_dir,paste(dataSetName,"_StormCounts.csv",sep="")))
}



#########################################
# PLOT AND SAVE DATA - EVENT SEPARATION #
#########################################

# Make subfolder in output directory to save hydrograph plots
dir.create(file.path(output_dir, "Hydrographs"), showWarnings = FALSE)

# 1) Plot and save the hydrograph with input data

initialHydrograph <- ggplot(allInputData30Min,aes(x=datetime, y=q_cms)) +
                            geom_line(size=0.5, color="black") +
                            xlab(NULL) +
                            ylab(expression(paste("Total discharge (",m^3," ",s^-1,")"))) +
                            theme_bw() +
                            theme(text=element_text(size=18))
if (select_year){
  figname <- paste(dataSetName, "_Year_", year_sel, "_TotalDischarge.jpeg")
}else{
  figname <- paste(dataSetName, "_TotalDischarge.jpeg")
}
ggsave(file=file.path(output_dir,"Hydrographs", figname),
       initialHydrograph,
       width = 12, 
       height = 4, 
       units = "in",
       dpi=600)


# 2) Plot total discharge with baseflow

baseflowHydrograph <- ggplot() + 
                            geom_line(data=batchRunFlowsLF1, aes(x=datetime, y=total_flow), size=0.5, color="black") +
                            geom_line(data=batchRunFlowsLF1, aes(x=datetime, y=base_flow,color=filter_para), size=0.75) +
                            scale_color_brewer(palette = "Set1") +
                            xlab(NULL) +
                            ylab(expression(paste("Discharge (",m^3," ",s^-1,")"))) +
                            theme_bw() +
                            theme(legend.title = element_blank(),
                                  text=element_text(size=18))
if (select_year){
  figname <- paste(dataSetName, "_Year_", year_sel, "_Baseflows.jpeg")
}else{
  figname <- paste(dataSetName, "_Baseflows.jpeg")
}

ggsave(file=file.path(output_dir,"Hydrographs", figname),
       baseflowHydrograph,
       width = 14, 
       height = 4, 
       units = "in",
       dpi=600)


# 3) Plot smoothed storm flows

stormflowHydrograph <- ggplot() + 
  geom_line(data=batchRunFlowsLF1, aes(x=datetime, y=storm_flow,color=filter_para), size=0.75) +
  scale_color_brewer(palette = "Set1") +
  xlab(NULL) +
  ylab(expression(paste("Storm flow (",m^3," ",s^-1,")"))) +
  theme_bw() +
  theme(legend.title = element_blank(),
        text=element_text(size=18))

# Set name of figs
if (select_year){
  figname <- paste(dataSetName, "_Year_", year_sel, "_StormflowsOnly.jpeg")
}else{
  figname <- paste(dataSetName, "_StormflowsOnly.jpeg")
}
ggsave(file=file.path(output_dir,"Hydrographs", figname),
       stormflowHydrograph,
       width = 14, 
       height = 4, 
       units = "in",
       dpi=600)


# 3a) Plot smoothed storm flows with storm flow thresholds

stormflowThreshHydrograph <- ggplot() + 
  geom_line(data=batchRunFlowsLF1, aes(x=datetime, y=storm_flow,color=filter_para), size=0.75) +
  scale_color_brewer(palette = "Set1") +
  geom_hline(yintercept = candidateSfThresh, linetype = "dashed", color = "black",alpha=0.5) +
  xlab(NULL) +
  ylab(expression(paste("Storm flow (",m^3," ",s^-1,")"))) +
  theme_bw() +
  theme(legend.title = element_blank(),
        text=element_text(size=18))

# Set name of figs
if (select_year){
  figname <- paste(dataSetName, "_Year_", year_sel, "_StormflowsOnlyWithThresholds.jpeg")
}else{
  figname <- paste(dataSetName, "_StormflowsOnlyWithThresholds.jpeg")
}
ggsave(file=file.path(output_dir,"Hydrographs",figname),
       stormflowThreshHydrograph,
       width = 14, 
       height = 4, 
       units = "in",
       dpi=600)


# 4) Plot batch run event separation hydrographs
eventsDataShaded1 <- eventsData1 %>% mutate(start = as.POSIXct(start,
                                                                   format("%Y-%m-%d %H:%M:%S"),tz="EST"),
                                            end = as.POSIXct(end,
                                                                   format("%Y-%m-%d %H:%M:%S"),tz="EST"),
                                            tops = max(allInputData30Min$q_cms),
                                            bottoms = 0)

batchEventSepPlot <- ggplot() + 
  geom_rect(data=eventsDataShaded1, mapping=aes(xmin=start, 
                                                xmax=end, 
                                                ymin=bottoms, 
                                                ymax=tops), fill="green", color="red", alpha=0.2) +
  
  geom_line(data=allInputData30Min, aes(x=datetime, y=q_cms), size=0.8, color="blue") +
  geom_line(data=allInputData30Min, aes(x=datetime, y=rescaled_conc), size=0.5, color="black",linetype="dashed") +
  facet_wrap(~ run_id, ncol = 1) +
  scale_color_brewer(palette = "Set1") +
  xlab(NULL) +
  ylab(expression(paste("Discharge (",m^3," ",s^-1,")"))) +
  theme_bw() +
  theme(legend.title = element_blank(),
        text=element_text(size=18))

# Set name of figs
if (select_year){
  figname <- paste(dataSetName, "_Year_", year_sel, "_BatchEventSeparationPlot.jpeg")
}else{
  figname <- paste(dataSetName, "_BatchEventSeparationPlot.jpeg")
}
ggsave(file=file.path(output_dir,"Hydrographs",figname),
       batchEventSepPlot,
       width = 14, 
       height = 15, 
       units = "in",
       dpi=600)

####################################
# PLOT AND SAVE DATA - c-Q RESULTS #
####################################

if (constit == "NO3") {
  
  makeCQPlotsNO3(batchRun1)
  makeHystFlushPlotsNO3(hysteresisData1)
  
} else if (constit == "TOC") {

makeCQPlotsTOC(batchRun1)
makeHystFlushPlotsTOC(hysteresisData1)
  
} else if (constit == "turbity") {
    
makeCQPlotsTurb(batchRun1) 
makeHystFlushPlotsTurb(hysteresisData1)
  
}

