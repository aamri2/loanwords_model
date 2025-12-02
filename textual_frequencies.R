library(tidyr)
library(dplyr)

freqs <- read.csv('textual_frequency_librispeechCL.csv')

p <- read.csv('probabilities/p_humans_wv.csv')
p <- read.csv('probabilities/p_w2v2_transformer_ctc_2_timit_30e_1_decode_vowels_wv.csv')
p <- p[p$language == 'EN',]
p <- pivot_wider(p, id_cols = file, names_from = classification, values_from = probabilities)

# add contexts to probabilities
human <- read.csv('../human_vowel_responses.csv')
contexts <- unique(human[human$language_stimuli=='EN',c("filename", "prev_phone", "X.phone", "next_phone")])
row.names(contexts) <- contexts$filename
p$prev_phone <- contexts[p$file,"prev_phone"]
p$next_phone <- contexts[p$file, "next_phone"]
p$vowel <- contexts[p$file, "X.phone"]

# add P(C_, _C|V) to probabilities
freqs[grepl("^_.", freqs$context),"P(.|V)"] <- freqs[grepl("^_.", freqs$context),] %>%
  group_by(vowel) %>% mutate("P(.|V)" = frequency/sum(frequency)) %>% .["P(.|V)"]
freqs[grepl("._$", freqs$context),"P(.|V)"] <- freqs[grepl("._$", freqs$context),] %>%
  group_by(vowel) %>% mutate("P(.|V)" = frequency/sum(frequency)) %>% .["P(.|V)"]
p$"P(C_|V)" <- freqs$"P(.|V)"[match(paste0(p$prev_phone, "_", p$vowel), paste0(freqs$context, freqs$vowel))]
p$"P(_C|V)" <- freqs$"P(.|V)"[match(paste0("_", p$next_phone, p$vowel), paste0(freqs$context, freqs$vowel))]
p <- p %>% mutate("P(C_, _C|V)" = .$"P(C_|V)" * .$"P(_C|V)")
p$correct <- p %>% rowwise() %>% mutate(correct = get(vowel)) %>% .$correct



freqs[grepl("^_.", freqs$context),"P(.|C_C)"] <- freqs[grepl("^_.", freqs$context),] %>%
  group_by(context) %>% mutate("P(.|C_C)" = frequency/sum(frequency)) %>% .["P(.|C_C)"]
freqs[grepl("._$", freqs$context),"P(.|C_C)"] <- freqs[grepl("._$", freqs$context),] %>%
  group_by(context) %>% mutate("P(.|C_C)" = frequency/sum(frequency)) %>% .["P(.|C_C)"]
p$"P(V|C_)" <- freqs$"P(.|C_C)"[match(paste0(p$prev_phone, "_", p$vowel), paste0(freqs$context, freqs$vowel))]
p$"P(V|_C)" <- freqs$"P(.|C_C)"[match(paste0("_", p$next_phone, p$vowel), paste0(freqs$context, freqs$vowel))]
p <- p %>% mutate("P(V|C_C)" = .$"P(V|C_)" * .$"P(V|_C)")

p$"P(V)" <- freqs$"P(.|V)"[match(paste0(p$prev_phone, "_", p$vowel), paste0(freqs$context, freqs$vowel))]
p$`P(V|C_C)` <- freqs$`P(V|C_C)`[match]

p <- p[p$correct > 0,] # !!!

library(ggplot2)

ggplot(p, mapping=aes(x = log(`P(C_, _C|V)`), y = log(correct))) + geom_point() + facet_wrap(~ vowel)
plot(log(correct) ~ log(`P(V|C_C)`), p)
abline(lm(log(p$correct) ~ log(p$`P(V|C_C)`)))
plot(correct ~ `P(C_, _C|V)`, p)
abline(lm(p$correct ~ p$`P(C_, _C|V)`))

ggplot(p, mapping=aes(x = log(`P(V|C_C)`), y = log(correct))) + geom_point() + facet_wrap(~ vowel)


# P(V|C_C) = P(C_C|V)P(V)/P(C_C)

# write out full reasoning P(C_C, V, S) ... to linear model

# stan modelling...

# P(CVC|S) = ...
# sum(P(CVC)) = P(V)
# sum_c_c(P(V|C_C)P(C_C)) = P(V)
# 
# P(C)

# use (Representation)JSD as loss? - should already handle 0's...
# y as softmaxed transformation... or direct calculation...
  # no normal, score from JSD...
  # log probs, softmax y...
library(rstan)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)
fit <- stan('textual_frequencies.stan', data = list(y=p$correct, x=p$`P(C_, _C|V)`, N=length((p$correct))))
