library(tidymodels)
library(ranger)
library(tidyverse)

data = read_tsv('mushrooms.tsv') %>%
  filter(is.na(Cap_Surface) | Cap_Surface != "15.94") %>%
  filter(is.na(Gill_Attachment) | Gill_Attachment != "32.54") %>%
  filter(is.na(Gill_Spacing) | Gill_Spacing != "2.69" | Gill_Spacing != "3.61") %>%
  filter(is.na(Gill_Color) | Gill_Color != "3.45" | Gill_Color != "5") %>%
  filter(is.na(Stem_Height) | Stem_Height != "0") %>%
  filter(is.na(Stem_Width) | Stem_Width != "0") %>%
  filter(is.na(Stem_Root) | Stem_Root != "2.77" | Stem_Root != "20.01" | Stem_Root != "5.59") %>%
  filter(is.na(Habitat) | Habitat != "17.1" | Habitat != "8.09" | Habitat != "habitat") %>%
  str()