library(ggplot2)
library(dplyr)
library(purrr)
library(export)

# CSV 파일 불러오기
data <- read.csv("~/Google Drive/My Drive/UNIST/continual_learning/KCC_2025/output/result_final_hop.csv")
# data <- read.csv("G:/내 드라이브/UNIST/continual_learning/KCC_2025/output/result.csv")
print(data)


# GCC 결과는 별도 파일에서 불러오기
gcc_only <- read.csv("~/Google Drive/My Drive/UNIST/continual_learning/KCC_2025/output/GCC.csv")

# sample_rate 목록 추출
unique_rates <- unique(data$sample_rate)

# GCN_sample 결과 확장: sample_rate == 1.0 결과를 모든 sample_rate에 복사
gcn_only <- data %>% filter(method == "GCN_sample", sample_rate == 1.0)

gcc_extended <- do.call(rbind, lapply(unique_rates, function(rate) {
  gcc_only %>% mutate(sample_rate = rate, hop = 2)
}))

gcn_extended <- do.call(rbind, lapply(unique_rates, function(rate) {
  gcn_only %>% mutate(sample_rate = rate, hop = 2)
}))


# GCC 외 데이터와 합치기
df_plot <- data %>%
  filter(method != "GCC") %>%
  filter(sample_rate != 1.0) %>%
  bind_rows(gcc_extended, gcn_extended)

# 시각화 함수: 막대그래프 (follower_computation)
plot_follower_computation_bar <- function(data, dataset_name) {
  filtered_data <- data %>%
    # filter(dataset == dataset_name) %>%
    filter(sample_rate == 0.5, method != 'GConvLSTM_combine') %>%
    # filter(dataset != "facebook") %>%
    arrange(method, budget)
  
  p <- ggplot(filtered_data, aes(x = budget, y = follower_computation, fill = method)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.8)) +
    facet_grid(dataset ~ sample_rate, labeller = label_both, scales = "free_y") +
    scale_x_continuous(breaks = seq(0, 10, by = 2))+
    labs(
      # title = paste0("Follower Computation by Budget"),
      x = "b",
      y = "Follower Computation",
      fill = "Method"
    ) +
    scale_fill_manual(
      values = c(
        "GCC" = "#e41a1c",
        "GCN_sample" = "#4daf4a",
        "GConvLSTM_combine" = "#00bfc4",
        "GConvLSTM_partial" = "#984ea3"
      ),
      labels = c(
        "GCC" = "GCC",
        "GCN_sample" = "GCN",
        "GConvLSTM_combine" = "GConvLSTM_Test_Train",
        "GConvLSTM_partial" = "GConvLSTM"
      )
    ) +
    theme_minimal() +
    theme(
      axis.title.x = element_text(size = 18),
      axis.title.y = element_text(size = 16),
      axis.text.x = element_text(size = 14),
      axis.text.y = element_text(size = 14),
      axis.ticks = element_line(color = "black", size = 0.6),              # ✅ 눈금선 켜기
      axis.ticks.length = unit(0.25, "cm"),                                # ✅ 눈금 길이 조정
      axis.line = element_line(color = "black", size = 0.6),               # ✅ 축선 강조
      panel.grid = element_blank(),                                        # ❌ 격자선 제거
      legend.title = element_text(size = 18),
      legend.text = element_text(size = 14),
      plot.title = element_text(size = 20, face = "bold"),
      strip.text = element_text(size = 14),
      panel.border = element_rect(color = "black", fill = NA, linewidth = 1)
    )
  
  print(p)
}

# 시각화 함수: 2x2 패널 버전
plot_follower_computation_bar_22 <- function(data) {
  filtered_data <- data %>%
    filter(sample_rate == 0.5, method != 'GConvLSTM_combine', hop == 2) %>%
    filter(dataset %in% c("football", "karate", "mexican", "strike")) %>%
    arrange(method, budget)
  
  p <- ggplot(filtered_data, aes(x = budget, y = follower_computation, fill = method)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.8)) +
    facet_wrap(~ dataset, ncol = 2, scales = "free_y") +
    scale_x_continuous(breaks = seq(0, 10, by = 2)) +
    labs(
      x = "b",
      y = "Follower Computation",
      fill = "Method"
    ) +
    scale_fill_manual(
      values = c(
        "GCC" = "#e41a1c",
        "GCN_sample" = "#4daf4a",
        "GConvLSTM_combine" = "#00bfc4",
        "GConvLSTM_partial" = "#984ea3"
      ),
      labels = c(
        "GCC" = "GCC",
        "GCN_sample" = "GCN",
        "GConvLSTM_combine" = "GConvLSTM_Test_Train",
        "GConvLSTM_partial" = "GConvLSTM"
      )
    ) +
    theme_minimal() +
    theme(
      axis.title.x = element_text(size = 18),
      axis.title.y = element_text(size = 16),
      axis.text.x = element_text(size = 14),
      axis.text.y = element_text(size = 14),
      axis.ticks = element_line(color = "black", size = 0.6),              # ✅ 눈금선 켜기
      axis.ticks.length = unit(0.25, "cm"),                                # ✅ 눈금 길이 조정
      axis.line = element_line(color = "black", size = 0.6),               # ✅ 축선 강조
      panel.grid = element_blank(),                                        # ❌ 격자선 제거
      legend.title = element_text(size = 18),
      legend.text = element_text(size = 14),
      plot.title = element_text(size = 20, face = "bold"),
      strip.text = element_text(size = 14),
      panel.border = element_rect(color = "black", fill = NA, linewidth = 1)
    )
  
  print(p)
}


# 예시 실행
plot_follower_computation_bar(df_plot, dataset_name = "karate")

# 실행
plot_follower_computation_bar_22(df_plot)
graph2ppt(file="~/Google Drive/My Drive/UNIST/continual_learning/KCC_2025/output/eq2.pptx")
