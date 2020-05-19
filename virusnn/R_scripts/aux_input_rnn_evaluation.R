options(width=150)
library(tidyverse)
library(Biostrings)

# Define infile
predprob_file = "results/2020-05-13/gisaid_cov2020.test_50.probabilistic_prediction_samples.tab"
predargm_file = "results/2020-05-13/gisaid_cov2020.test_50.argmax_prediction_samples.tab"
loss_file = "results/2020-05-13/aux_input_rnn_loss.tab.gz"

# Load data
predictions = bind_rows(
  read_tsv(predprob_file, na=character()) %>% mutate(Prediction = "Sampling"),
  read_tsv(predargm_file, na=character()) %>% mutate(Prediction = "Argmax")
)
loss = read_tsv(loss_file, col_names=c("Sample", "Loss"))
loss = loss %>%
  mutate(Update = Sample/256) %>%
  filter(Sample == 256) %>%
  mutate(Model = 1:nrow(.)) %>%
  filter(Model <= 370) %>%
  mutate(Update = Update + (Model-1) * 64 * 36314 / 256)

# Calculate percent identity
predictions = predictions %>% mutate(
  DescAncId=pid(pairwiseAlignment(Ancestral, Descendant), "PID4"),
  PredDescId=pid(pairwiseAlignment(Descendant, Predicted), "PID4"),
  PredAncId=pid(pairwiseAlignment(Ancestral, Predicted), "PID4"),
  AncLen=nchar(Ancestral),
  DescLen=nchar(Descendant),
  PredLen=nchar(Predicted)
) %>% mutate(
  DescAncId = case_when(
    DescLen == 0 & AncLen == 0 ~ 100,
    DescLen != AncLen & (DescLen == 0 | AncLen == 0) ~ 0,
    T ~ DescAncId
  ),
  PredDescId = case_when(
    DescLen == 0 & PredLen == 0 ~ 100,
    DescLen != PredLen & (DescLen == 0 | PredLen == 0) ~ 0,
    T ~ PredDescId
  ),
  PredAncId = case_when(
    AncLen == 0 & PredLen == 0 ~ 100,
    AncLen != PredLen & (AncLen == 0 | PredLen == 0) ~ 0,
    T ~ PredAncId
  ),
  Relation = 1:nrow(.)
)

predplot = predictions %>%
  select(-Ancestral, -Descendant, -Predicted) %>%
  gather(
    Data, Value,
    -Relation, -Position, -Hotspot, -Distance, -Prediction
  ) %>%
  mutate(Value = ifelse(endsWith(Data, "Id"), (100 - Value)/100, Value)) %>%
  mutate(
    Data = case_when(
      Data == "DescAncId" ~ "ExpectedMutation",
      Data == "PredAncId" ~ "PredictedMutation",
      Data == "PredDescId" ~ "Error",
      T ~ Data
    )
  ) %>%
  spread(Data, Value) %>%
#  filter(AncLen > 0) %>%
  mutate(Hotspot = ifelse(Hotspot == 1, "Hotspot", "No hotspot"))

# Plot errors
gp = ggplot(
  predplot,
  aes(x=ExpectedMutation, y=PredictedMutation, colour=Error)
)
gp = gp + geom_point(alpha=0.6)
gp = gp + theme_bw()
gp = gp + scale_colour_distiller(palette='PuRd', direction=1)
gp = gp + facet_grid(Prediction~Hotspot)
gp = gp + theme(
  aspect.ratio=1,
  axis.text = element_text(colour="black"),
  axis.ticks = element_line(colour="black"),
  strip.background = element_blank(),
  axis.text.x = element_text(angle=30, hjust=0.9, vjust=0.8)
)
gp = gp + ylab("Predicted mutation") + xlab("Expected mutation")
gp1 = gp

# Plot amount of predicted mutation vs distance
gp = ggplot(predplot, aes(x=Distance, y=PredictedMutation))
gp = gp + geom_hex(bins=15)
gp = gp + theme_bw()
gp = gp + facet_grid(Prediction~Hotspot)
gp = gp + scale_fill_distiller(palette='BuGn', direction=1)
gp = gp + theme(
  aspect.ratio=1,
  axis.text = element_text(colour="black"),
  axis.ticks = element_line(colour="black"),
  strip.background = element_blank(),
  axis.text.x = element_text(angle=30, hjust=0.9, vjust=0.8)
)
gp = gp + ylab("Predicted mutation")
gp2 = gp

# Plot loss
gp = ggplot(loss, aes(x=Update, y=Loss))
gp = gp + geom_line()
gp = gp + theme_bw()
gp = gp + scale_y_log10()
gp = gp + annotation_logticks(sides='l')
gp = gp + theme(
  axis.text = element_text(colour="black"),
  axis.ticks = element_line(colour="black"),
  axis.text.x = element_text(angle=30, hjust=0.9, vjust=0.8)
)
gp3 = gp

# Plot predicted lengths
predlen = predplot %>%
  select(Prediction, Relation, AncLen, DescLen, PredLen) %>%
  gather(Sequence, Length, -Prediction, -Relation, -AncLen) %>%
  mutate(
    Sequence = ifelse(Sequence == "DescLen", "Expected", "Predicted"),
    Group = paste(Sequence, AncLen)
  )

gp = ggplot(
  predlen, aes(group=Group, colour=Sequence, x=AncLen, y=Length)
)
gp = gp + geom_boxplot(outlier.size=0.5)
gp = gp + theme_bw()
gp = gp + facet_grid(~Prediction)
gp = gp + scale_x_continuous(breaks=1:16)
gp = gp + scale_y_continuous(breaks=0:17)
gp = gp + scale_colour_manual(values=c("#998ec3","#f1a340"))
gp = gp + theme(
  axis.text = element_text(colour="black"),
  axis.ticks = element_line(colour="black"),
  strip.background = element_blank()
)
gp = gp + xlab("Ancestral sequence length")
gp4 = gp

# Arrange plots
library(ggpubr)
outfile = "art/2020-05-15/aux_input_rnn_evaluation.2.pdf"
pdf(outfile, width=9, height=6, onefile=FALSE)
ggarrange(
#  gp3,
  ggarrange(
    gp1, gp2,
    ncol=2,
    labels = c("A","B")
  ),
  gp4,
  nrow = 2,
  ncol = 1,
  labels = c("","C"),
  heights = c(1.2, 1)
)
garbage = dev.off()
