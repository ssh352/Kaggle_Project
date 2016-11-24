library(caret)

result <- read.csv("stack/test_encode2.csv")
e1_rf <- read.csv("stack/proc_s_RF_t.csv")
e1_xb <- read.csv("stack/proc_s_XGB2_t.csv")

df <- data.frame(loss = result$loss,
                 e1_rf = e1_rf$loss,
                 e1_xb = e1_xb$loss)

mae_e1_rf <- mean(abs(df$loss - df$e1_rf))   ### 1232.126
mae_e1_xb <- mean(abs(df$loss - df$e1_xb))   ### 1228.839

stack_model <- train(loss ~ ., method = "gam", data = df)
pred <- predict(stack_model, df)
mean(abs(df$loss - pred))

# predict submit
submission <- read.csv("Prediction/sample_submission.csv")
e1_rf_s <- read.csv("Prediction/proc_RF_submit_1263_ntree1000.csv")
e1_xb_s <- read.csv("Prediction/proc_XGB2.csv")

df_s <- data.frame(e1_rf = e1_rf_s$loss, 
                   e1_xb = e1_xb_s$loss)
pred_s <- predict(stack_model, df_s)

submission$loss <- as.numeric(pred_s)

write.csv(submission, file = 'submission1123.csv', row.names = F)
