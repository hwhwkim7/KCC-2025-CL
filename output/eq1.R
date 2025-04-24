library(ggplot2)
library(dplyr)
library(purrr)
library(export)

# CSV 파일 불러오기
data <- read.csv("~/Google Drive/My Drive/UNIST/continual_learning/KCC_2025/output/result_final_hop.csv")
print(data)

# GCC 결과 확장: sample_rate == 1.0 결과를 모든 sample_rate에 복사
gcc_only <- read.csv("~/Google Drive/My Drive/UNIST/continual_learning/KCC_2025/output/GCC.csv")
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

# 시각화 함수 정의 (dataset과 y축 범위 조절 가능)
plot_cumulative_eval_time_facet <- function(data, dataset_name, ymin = NULL, ymax = NULL) {
  filtered_data <- data %>%
    filter(sample_rate != 1.0) %>%
    filter(dataset %in% c("football", "karate", "mexican", "strike")) %>%
    mutate(method_hop = paste0(method, "_hop", hop)) %>%  # ✅ method-hop 조합 생성
    arrange(sample_rate, method_hop, budget) %>%
    group_by(sample_rate, method_hop) %>%
    ungroup()
  
  p <- ggplot(filtered_data, aes(x = budget, y = coreness_loss,
                                 color = method_hop,
                                 shape = method_hop,
                                 fill = method_hop,
                                 group = method_hop)) +
    geom_line(size = 1.2) +
    geom_point(
      size = 2,
      stroke = 1.5,
      color = "black"
    ) +
    scale_color_manual(
      values = c(
        "GCC_hopNA" = "#e41a1c",              # 빨강
        "GCN_sample_hop0" = "#4daf4a",        # 녹색
        "GConvLSTM_partial_hop1" = "#377eb8", # 파랑
        "GConvLSTM_partial_hop2" = "#984ea3"  # 보라
      ),
      labels = c("GCC", "GCN", "GConvLSTM_hop1", "GConvLSTM_hop2")
      
    ) +
    
    scale_shape_manual(
      values = c(
        "GCC_hopNA" = 21,                    # ●
        "GCN_sample_hop0" = 24,              # ▲
        "GConvLSTM_partial_hop1" = 22,       # ■
        "GConvLSTM_partial_hop2" = 23        # ◆ 다이아몬드
      ),
      labels = c("GCC", "GCN", "GConvLSTM_hop1", "GConvLSTM_hop2")
      
    ) +
    
    scale_fill_manual(
      values = c(
        "GCC_hopNA" = "#e41a1c",
        "GCN_sample_hop0" = "#4daf4a",
        "GConvLSTM_partial_hop1" = "#377eb8",
        "GConvLSTM_partial_hop2" = "#984ea3"
      ),
      labels = c("GCC", "GCN", "GConvLSTM_hop1", "GConvLSTM_hop2")
      
    )+
    scale_x_continuous(breaks = seq(0, 10, by = 2)) +
    facet_grid(dataset ~ sample_rate, labeller = label_both, scales = "free_y") +
    labs(
      x = "budget",
      y = "Coreness loss",
      color = "Method+Hop",
      shape = "Method+Hop",
      fill = "Method+Hop"
    ) +
    theme_minimal() +
    theme(
      axis.title.x = element_text(size = 18),
      axis.title.y = element_text(size = 16),
      axis.text.x = element_text(size = 14),
      axis.text.y = element_text(size = 14),
      axis.ticks = element_line(color = "black", size = 0.6),
      axis.ticks.length = unit(0.25, "cm"),
      axis.line = element_line(color = "black", size = 0.6),
      panel.grid = element_blank(),
      legend.title = element_text(size = 18),
      legend.text = element_text(size = 12),
      plot.title = element_text(size = 20, face = "bold"),
      strip.text = element_text(size = 14),
      panel.border = element_rect(color = "black", fill = NA, linewidth = 1)
    )
  
  if (!is.null(ymin) && !is.null(ymax)) {
    p <- p + coord_cartesian(ylim = c(ymin, ymax))
  }
  
  print(p)
}

# 시각화 함수 정의 (dataset과 y축 범위 조절 가능)
plot_cumulative_eval_time_facet22 <- function(data, dataset_name, ymin = NULL, ymax = NULL) {
  filtered_data <- data %>%
    # filter(dataset != "facebook") %>%
    filter(sample_rate == 0.5, method != 'GConvLSTM_combine', hop == 2) %>%
    filter(dataset %in% c("football", "karate", "mexican", "strike")) %>%
    arrange(sample_rate, method, budget) %>%
    group_by(sample_rate, method) %>%
    ungroup()
  print(filtered_data)

  p <- ggplot(filtered_data, aes(x = budget, y = coreness_loss, group = method)) +
    scale_x_continuous(breaks = seq(0, 10, by = 2)) +
    geom_line(aes(color = method), size = 1.2) +
    geom_point(
      aes(fill = method, shape = method),  # 내부 채움 + 모양만 범례에 사용
      size = 4,
      stroke = 1.2,
      color = "black"                      # ✅ 테두리 고정
    ) +
    scale_fill_manual(
      values = c(
        "GCC" = "#e41a1c",
        "GCN_sample" = "#4daf4a",
        "GConvLSTM_partial" = "#984ea3"
      ),
      labels = c(
        "GCC" = "GCC",
        "GCN_sample" = "GCN",
        "GConvLSTM_partial" = "GConvLSTM"
      )
    ) +
    scale_color_manual(
      values = c(
        "GCC" = "#e41a1c",
        "GCN_sample" = "#4daf4a",
        "GConvLSTM_partial" = "#984ea3"
      ),
      guide = "none"  # ✅ 선 색은 범례에 안 넣기 (중복 제거)
    ) +
    scale_shape_manual(
      values = c(
        "GCC" = 21,             # ●
        "GCN_sample" = 24,      # ▲
        "GConvLSTM_partial" = 22  # ■
      ),
      labels = c(
        "GCC" = "GCC",
        "GCN_sample" = "GCN",
        "GConvLSTM_partial" = "GConvLSTM"
      )
    ) +
    labs(
      x = "budget",
      y = "Coreness loss",
      fill = "Method",       # ✅ fill이 legend 이름으로 쓰임
      shape = "Method"
    ) +
    facet_wrap(~ dataset, ncol = 2, scales = "free_y") +
    theme_minimal() +
    theme(
      axis.title.x = element_text(size = 18),
      axis.title.y = element_text(size = 16),
      axis.text.x = element_text(size = 25),
      axis.text.y = element_text(size = 25),
      axis.ticks = element_line(color = "black", size = 0.6),
      axis.ticks.length = unit(0.25, "cm"),
      axis.line = element_line(color = "black", size = 0.6),
      panel.grid = element_blank(),
      legend.title = element_text(size = 18),
      legend.text = element_text(size = 14),
      plot.title = element_text(size = 20, face = "bold"),
      strip.text = element_text(size = 14),
      panel.border = element_rect(color = "black", fill = NA, linewidth = 1)
    )
  
  if (!is.null(ymin) && !is.null(ymax)) {
    p <- p + coord_cartesian(ylim = c(ymin, ymax))
  }
  
  print(p)
}


# plot_cumulative_eval_time_facet(df_plot, dataset_name = "football")

plot_cumulative_eval_time_facet22(df_plot, dataset_name = "football")
graph2ppt(file="~/Google Drive/My Drive/UNIST/continual_learning/KCC_2025/output/eq1.pptx", width=15.5, height=10)
