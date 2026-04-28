library(dplyr)
library(xgboost)
library(ggplot2)
library(forcats)
library(Matrix)

df = read.csv('~/Downloads/all_screens (2).csv') # File created by Ryan that contains all the screen data for our analysis

players_for_names <- read.csv("/Users/anirudhkaranam/Downloads/sports-project/2025/players_involved_in_screen.csv")

screener_name_map <- players_for_names %>%
  select(screener_player_id, screener_name) %>%
  filter(!is.na(screener_player_id), !is.na(screener_name)) %>%
  distinct(screener_player_id, .keep_all = TRUE) %>%
  mutate(screener_player_id = as.character(screener_player_id)) %>%
  tibble::deframe()

scrd_name_map <- players_for_names %>%
  select(screener_defender_player_id, screener_defender_name) %>%
  filter(!is.na(screener_defender_player_id), !is.na(screener_defender_name)) %>%
  distinct(screener_defender_player_id, .keep_all = TRUE) %>%
  mutate(screener_defender_player_id = as.character(screener_defender_player_id)) %>%
  tibble::deframe()

bh_name_map <- players_for_names %>%
  select(ball_handler_player_id, ball_handler_name) %>%
  filter(!is.na(ball_handler_player_id), !is.na(ball_handler_name)) %>%
  distinct(ball_handler_player_id, .keep_all = TRUE) %>%
  mutate(ball_handler_player_id = as.character(ball_handler_player_id)) %>%
  tibble::deframe()

bhd_name_map <- players_for_names %>%
  select(ball_handler_defender_player_id, ball_handler_defender_name) %>%
  filter(!is.na(ball_handler_defender_player_id), !is.na(ball_handler_defender_name)) %>%
  distinct(ball_handler_defender_player_id, .keep_all = TRUE) %>%
  mutate(ball_handler_defender_player_id = as.character(ball_handler_defender_player_id)) %>%
  tibble::deframe()

TARGET <- "points"
TEST_SIZE <- 0.20
RANDOM_STATE <- 42
MIN_PLAYS <- 75
N_BOOT <- 200

set.seed(RANDOM_STATE)

screen_df <- df %>%
  filter(
    shooter_id == screen_bh | shooter_id == screener_id
  ) %>%
  mutate(
    screener_id = as.character(screener_id)
  )

feature_cols <- c(
  "screen_type",
  "screen_location",
  "screen_bhd",
  "screen_scrd",
  "screen_defense",
  "shot_distance",
  "shot_type",
  "shot_contest_type"
)

cat_cols <- c(
  "screen_type",
  "screen_location",
  "screen_bhd",
  "screen_scrd",
  "screen_defense",
  "shot_type",
  "shot_contest_type"
)

keep_cols <- c(feature_cols, TARGET, "screener_id", "screen_bh")

model_df <- screen_df %>%
  select(all_of(keep_cols)) %>%
  tidyr::drop_na()

model_df <- model_df %>%
  mutate(across(all_of(cat_cols), as.factor))

n <- nrow(model_df)
test_idx <- sample(seq_len(n), size = floor(TEST_SIZE * n))
train_idx <- setdiff(seq_len(n), test_idx)

train_df <- model_df[train_idx, ]
test_df  <- model_df[test_idx, ]

X_train <- train_df %>% select(all_of(feature_cols))
X_test  <- test_df %>% select(all_of(feature_cols))
y_train <- train_df[[TARGET]]
y_test  <- test_df[[TARGET]]

train_matrix <- sparse.model.matrix(
  as.formula(paste("~", paste(feature_cols, collapse = " + "), "-1")),
  data = X_train
)
test_matrix <- sparse.model.matrix(
  as.formula(paste("~", paste(feature_cols, collapse = " + "), "-1")),
  data = X_test
)

full_matrix <- sparse.model.matrix(
  as.formula(paste("~", paste(feature_cols, collapse = " + "), "-1")),
  data = model_df %>% select(all_of(feature_cols))
)

dtrain <- xgb.DMatrix(data = train_matrix, label = y_train)
dtest  <- xgb.DMatrix(data = test_matrix, label = y_test)
dfull  <- xgb.DMatrix(data = full_matrix, label = model_df[[TARGET]])

params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  eta = 0.03,
  max_depth = 4,
  subsample = 0.8,
  colsample_bytree = 0.7,
  colsample_bylevel = 0.7,
  min_child_weight = 10,
  alpha = 0.1,
  lambda = 2.0
)

watchlist <- list(train = dtrain, eval = dtest)

reg <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 1500,
  watchlist = watchlist,
  early_stopping_rounds = 50,
  print_every_n = 100,
  verbose = 1
)

pred_test <- predict(reg, dtest)
rmse <- sqrt(mean((y_test - pred_test)^2))
mae <- mean(abs(y_test - pred_test))

cat("Rows used:", format(nrow(model_df), big.mark = ","), "\n")
cat(sprintf("Test RMSE: %.4f\n", rmse))
cat(sprintf("Test MAE:  %.4f\n", mae))
cat("Best iteration:", reg$best_iteration, "\n")

model_df <- model_df %>%
  mutate(
    context_pred = predict(reg, dfull),
    residual = .data[[TARGET]] - context_pred
  )

screener_summary <- model_df %>%
  group_by(screener_id) %>%
  summarise(
    n_plays = n(),
    mean_residual = mean(residual),
    mean_points = mean(.data[[TARGET]]),
    mean_context_pred = mean(context_pred),
    .groups = "drop"
  ) %>%
  filter(n_plays >= MIN_PLAYS) %>%
  arrange(desc(mean_residual))

if (exists("screener_name_map")) {
  screener_summary <- screener_summary %>%
    mutate(screener_name = screener_name_map[screener_id]) %>%
    relocate(screener_name)
} else {
  screener_summary <- screener_summary %>%
    mutate(screener_name = screener_id) %>%
    relocate(screener_name)
}

print(head(screener_summary, 20))

boot_results <- vector("list", N_BOOT)

for (b in seq_len(N_BOOT)) {
  set.seed(RANDOM_STATE + b)
  
  sample_idx <- sample(seq_len(nrow(model_df)), size = nrow(model_df), replace = TRUE)
  sample_df <- model_df[sample_idx, ]
  
  boot_grouped <- sample_df %>%
    group_by(screener_id) %>%
    summarise(
      n_plays = n(),
      mean_residual = mean(residual),
      .groups = "drop"
    ) %>%
    filter(n_plays >= MIN_PLAYS) %>%
    select(screener_id, mean_residual)
  
  names(boot_grouped)[2] <- paste0("boot_", b)
  boot_results[[b]] <- boot_grouped
}

boot_df <- Reduce(function(x, y) full_join(x, y, by = "screener_id"), boot_results)

boot_long <- boot_df %>%
  tidyr::pivot_longer(
    cols = starts_with("boot_"),
    names_to = "bootstrap",
    values_to = "value"
  )

boot_summary <- boot_long %>%
  group_by(screener_id) %>%
  summarise(
    boot_mean = mean(value, na.rm = TRUE),
    boot_std = sd(value, na.rm = TRUE),
    ci_lower = quantile(value, 0.025, na.rm = TRUE),
    ci_upper = quantile(value, 0.975, na.rm = TRUE),
    .groups = "drop"
  )

screener_summary <- screener_summary %>%
  left_join(boot_summary, by = "screener_id") %>%
  arrange(desc(mean_residual))

print(head(screener_summary, 20))

importance <- xgb.importance(model = reg)

top_n <- 10

importance_top <- importance %>%
  arrange(desc(Gain)) %>%
  slice_head(n = top_n)

ggplot(importance_top, aes(x = Gain, y = fct_reorder(Feature, Gain))) +
  geom_col() +
  labs(
    title = "Top 10 xGBoost feature importances",
    x = "Gain",
    y = "Feature"
  ) +
  theme_minimal()

# Note: I used ChatGPT to come up with a theme for the plots
court_theme <- function() {
  theme_minimal(base_family = "serif") +
    theme(
      plot.title        = element_text(size = 14, face = "bold", margin = margin(b = 4)),
      plot.subtitle     = element_text(size = 10, color = "#666666", margin = margin(b = 12)),
      plot.caption      = element_text(size = 8, color = "#999999", margin = margin(t = 10)),
      plot.background   = element_rect(fill = "#F7F7F5", color = NA),
      panel.background  = element_rect(fill = "#F7F7F5", color = NA),
      panel.grid.major.x = element_line(color = "#E0E0DA", linewidth = 0.4),
      panel.grid.major.y = element_blank(),
      panel.grid.minor  = element_blank(),
      axis.text.y       = element_text(size = 10, color = "#222222"),
      axis.text.x       = element_text(size = 9, color = "#555555"),
      axis.title.x      = element_text(size = 10, color = "#444444", margin = margin(t = 8)),
      axis.title.y      = element_blank(),
      plot.margin       = margin(16, 20, 12, 12)
    )
}

positive_col <- "#2166AC"
negative_col <- "#D6604D"


n_show <- 10

top_players <- screener_summary %>% slice_head(n = n_show)
bottom_players <- screener_summary %>% slice_tail(n = n_show) %>% arrange(mean_residual)

plot_df <- bind_rows(top_players, bottom_players) %>%
  mutate(
    screener_name = as.character(screener_name),
    screener_name = fct_reorder(screener_name, mean_residual),
    color = ifelse(mean_residual >= 0, positive_col, negative_col)
  )

ggplot(plot_df, aes(x = mean_residual, y = screener_name)) +
  geom_vline(xintercept = 0, linetype = "solid", color = "#AAAAAA", linewidth = 0.6) +
  geom_errorbarh(
    aes(xmin = ci_lower, xmax = ci_upper),
    height = 0, linewidth = 0.5, color = "#BBBBBB"
  ) +
  geom_point(aes(color = color), size = 3.5) +
  scale_color_identity() +
  labs(
    title    = "Screener Impact: Points \nAbove/Below Context Expectation",
    subtitle = "Top 10 and bottom 10 screeners · 95% bootstrap intervals · min. 75 plays",
    x        = "Avg. residual (actual pts - context-predicted pts)",
    caption  = "Positive = screener generated more points than context predicts"
  ) +
  court_theme()

defender_summary <- model_df %>%
  group_by(screen_scrd) %>%
  summarise(
    n_plays = n(),
    mean_residual = mean(residual),
    mean_points = mean(.data[[TARGET]]),
    mean_context_pred = mean(context_pred),
    .groups = "drop"
  ) %>%
  filter(n_plays >= MIN_PLAYS) %>%
  arrange(desc(mean_residual))
defender_summary <- defender_summary %>%
  mutate(defender_name = scrd_name_map[as.character(screen_scrd)]) %>%
  relocate(defender_name)

boot_results_def <- vector("list", N_BOOT)

for (b in seq_len(N_BOOT)) {
  set.seed(RANDOM_STATE + b)
  
  sample_idx <- sample(seq_len(nrow(model_df)), size = nrow(model_df), replace = TRUE)
  sample_df <- model_df[sample_idx, ]
  
  boot_grouped_def <- sample_df %>%
    group_by(screen_scrd) %>%
    summarise(
      n_plays = n(),
      mean_residual = mean(residual),
      .groups = "drop"
    ) %>%
    filter(n_plays >= MIN_PLAYS) %>%
    select(screen_scrd, mean_residual)
  
  names(boot_grouped_def)[2] <- paste0("boot_", b)
  boot_results_def[[b]] <- boot_grouped_def
}

boot_df_def <- Reduce(function(x, y) full_join(x, y, by = "screen_scrd"), boot_results_def)

boot_long_def <- boot_df_def %>%
  tidyr::pivot_longer(
    cols = starts_with("boot_"),
    names_to = "bootstrap",
    values_to = "value"
  )

boot_summary_def <- boot_long_def %>%
  group_by(screen_scrd) %>%
  summarise(
    boot_mean = mean(value, na.rm = TRUE),
    boot_std = sd(value, na.rm = TRUE),
    ci_lower = quantile(value, 0.025, na.rm = TRUE),
    ci_upper = quantile(value, 0.975, na.rm = TRUE),
    .groups = "drop"
  )

defender_summary <- defender_summary %>%
  left_join(boot_summary_def, by = "screen_scrd") %>%
  arrange(desc(mean_residual))

top_defenders <- defender_summary %>%
  slice_head(n = n_show)

bottom_defenders <- defender_summary %>%
  slice_tail(n = n_show) %>%
  arrange(mean_residual)

plot_df_def <- bind_rows(top_defenders, bottom_defenders) %>%
  mutate(
    defender_name = as.character(defender_name),
    defender_name = ifelse(is.na(defender_name), as.character(screen_scrd), defender_name),
    defender_name = fct_reorder(defender_name, mean_residual),
    color = ifelse(mean_residual >= 0, positive_col, negative_col)
  )

ggplot(plot_df_def, aes(x = mean_residual, y = defender_name)) +
  geom_vline(xintercept = 0, linetype = "solid", color = "#AAAAAA", linewidth = 0.6) +
  geom_errorbarh(
    aes(xmin = ci_lower, xmax = ci_upper),
    height = 0, linewidth = 0.5, color = "#BBBBBB"
  ) +
  geom_point(aes(color = color), size = 3.5) +
  scale_color_identity() +
  labs(
    title    = "Screener Defender Impact: Points \nAllowed Above/Below Context Expectation",
    subtitle = "Top 10 and bottom 10 screener defenders · 95% bootstrap intervals · min. 75 plays",
    x        = "Avg. residual (actual pts - context-predicted pts)",
    caption  = "Positive = offense scored more points than context predicts"
  ) +
  court_theme()

bh_summary <- model_df %>%
  group_by(screen_bh) %>%
  summarise(
    n_plays = n(),
    mean_residual = mean(residual),
    mean_points = mean(.data[[TARGET]]),
    mean_context_pred = mean(context_pred),
    .groups = "drop"
  ) %>%
  filter(n_plays >= MIN_PLAYS) %>%
  arrange(desc(mean_residual)) %>%
  mutate(bh_name = bh_name_map[as.character(screen_bh)]) %>%
  relocate(bh_name)

boot_results_bh <- vector("list", N_BOOT)

for (b in seq_len(N_BOOT)) {
  set.seed(RANDOM_STATE + b)
  
  sample_idx <- sample(seq_len(nrow(model_df)), size = nrow(model_df), replace = TRUE)
  sample_df <- model_df[sample_idx, ]
  
  boot_grouped_bh <- sample_df %>%
    group_by(screen_bh) %>%
    summarise(
      n_plays = n(),
      mean_residual = mean(residual),
      .groups = "drop"
    ) %>%
    filter(n_plays >= MIN_PLAYS) %>%
    select(screen_bh, mean_residual)
  
  names(boot_grouped_bh)[2] <- paste0("boot_", b)
  boot_results_bh[[b]] <- boot_grouped_bh
}

boot_df_bh <- Reduce(function(x, y) full_join(x, y, by = "screen_bh"), boot_results_bh)

boot_long_bh <- boot_df_bh %>%
  tidyr::pivot_longer(
    cols = starts_with("boot_"),
    names_to = "bootstrap",
    values_to = "value"
  )

boot_summary_bh <- boot_long_bh %>%
  group_by(screen_bh) %>%
  summarise(
    boot_mean = mean(value, na.rm = TRUE),
    boot_std = sd(value, na.rm = TRUE),
    ci_lower = quantile(value, 0.025, na.rm = TRUE),
    ci_upper = quantile(value, 0.975, na.rm = TRUE),
    .groups = "drop"
  )

bh_summary <- bh_summary %>%
  left_join(boot_summary_bh, by = "screen_bh") %>%
  arrange(desc(mean_residual))

top_bh    <- bh_summary %>% slice_head(n = n_show)
bottom_bh <- bh_summary %>% slice_tail(n = n_show) %>% arrange(mean_residual)

plot_df_bh <- bind_rows(top_bh, bottom_bh) %>%
  mutate(
    bh_name = as.character(bh_name),
    bh_name = fct_reorder(bh_name, mean_residual),
    color = ifelse(mean_residual >= 0, positive_col, negative_col)
  )

ggplot(plot_df_bh, aes(x = mean_residual, y = bh_name)) +
  geom_vline(xintercept = 0, linetype = "solid", color = "#AAAAAA", linewidth = 0.6) +
  geom_errorbarh(
    aes(xmin = ci_lower, xmax = ci_upper),
    height = 0, linewidth = 0.5, color = "#BBBBBB"
  ) +
  geom_point(aes(color = color), size = 3.5) +
  scale_color_identity() +
  labs(
    title    = "Ball Handler Impact: Points \nAbove/Below Context Expectation",
    subtitle = "Top 10 and bottom 10 ball handlers · 95% bootstrap intervals · min. 75 plays",
    x        = "Avg. residual (actual pts - context-predicted pts)",
    caption  = "Positive = ball handler generated more points than context predicts"
  ) +
  court_theme()


bhd_summary <- model_df %>%
  group_by(screen_bhd) %>%
  summarise(
    n_plays = n(),
    mean_residual = mean(residual),
    mean_points = mean(.data[[TARGET]]),
    mean_context_pred = mean(context_pred),
    .groups = "drop"
  ) %>%
  filter(n_plays >= MIN_PLAYS) %>%
  arrange(desc(mean_residual))

boot_results_bhd <- vector("list", N_BOOT)

for (b in seq_len(N_BOOT)) {
  set.seed(RANDOM_STATE + b)
  
  sample_idx <- sample(seq_len(nrow(model_df)), size = nrow(model_df), replace = TRUE)
  sample_df <- model_df[sample_idx, ]
  
  boot_grouped_bhd <- sample_df %>%
    group_by(screen_bhd) %>%
    summarise(
      n_plays = n(),
      mean_residual = mean(residual),
      .groups = "drop"
    ) %>%
    filter(n_plays >= MIN_PLAYS) %>%
    select(screen_bhd, mean_residual)
  
  names(boot_grouped_bhd)[2] <- paste0("boot_", b)
  boot_results_bhd[[b]] <- boot_grouped_bhd
}

boot_df_bhd <- Reduce(function(x, y) full_join(x, y, by = "screen_bhd"), boot_results_bhd)

boot_summary_bhd <- boot_df_bhd %>%
  tidyr::pivot_longer(
    cols = starts_with("boot_"),
    names_to = "bootstrap",
    values_to = "value"
  ) %>%
  group_by(screen_bhd) %>%
  summarise(
    boot_mean = mean(value, na.rm = TRUE),
    boot_std = sd(value, na.rm = TRUE),
    ci_lower = quantile(value, 0.025, na.rm = TRUE),
    ci_upper = quantile(value, 0.975, na.rm = TRUE),
    .groups = "drop"
  )

bhd_summary <- bhd_summary %>%
  left_join(boot_summary_bhd, by = "screen_bhd") %>%
  arrange(desc(mean_residual))

top_bhd <- bhd_summary %>% slice_head(n = n_show)
bottom_bhd <- bhd_summary %>% slice_tail(n = n_show) %>% arrange(mean_residual)

plot_df_bhd <- bind_rows(top_bhd, bottom_bhd) %>%
  mutate(
    defender_name = bhd_name_map[as.character(screen_bhd)],
    defender_name = ifelse(is.na(defender_name),
                           as.character(screen_bhd),
                           defender_name),
    defender_name = fct_reorder(defender_name, mean_residual),
    color = ifelse(mean_residual >= 0, positive_col, negative_col)
  )

ggplot(plot_df_bhd, aes(x = mean_residual, y = defender_name)) +
  geom_vline(xintercept = 0, linetype = "solid", color = "#AAAAAA", linewidth = 0.6) +
  geom_errorbarh(
    aes(xmin = ci_lower, xmax = ci_upper),
    height = 0, linewidth = 0.5, color = "#BBBBBB"
  ) +
  geom_point(aes(color = color), size = 3.5) +
  scale_color_identity() +
  labs(
    title    = "Ball Handler Defender Impact: Points \nAllowed Above/Below Context Expectation",
    subtitle = "Top 10 and bottom 10 ball handler defenders · 95% bootstrap intervals · min. 75 plays",
    x        = "Avg. residual (actual pts - context-predicted pts)",
    caption  = "Positive = offense scored more points than context predicts"
  ) +
  court_theme()