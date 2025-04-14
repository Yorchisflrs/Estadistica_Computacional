# Paquetes necesarios
library(shiny)
library(readxl)
library(DT)
library(dplyr)
library(ggplot2)
library(shinythemes)
library(writexl)
library(openxlsx)
library(DescTools)
library(rmarkdown)
library(shinyjs)

# Interfaz de usuario (UI)
ui <- fluidPage(
  theme = shinytheme("cerulean"),
  useShinyjs(),
  titlePanel("APLICACIÓN DE ESTADÍSTICA: PRUEBA T Y ANOVA"),
  
  sidebarLayout(
    sidebarPanel(
      fileInput("file", "Sube tu archivo (.csv o .xlsx)",
                accept = c(".csv", ".xlsx")),
      uiOutput("varSelect"),
      actionButton("analyze", "Analizar", icon = icon("play")),
      conditionalPanel(
        condition = "input.testType == 't_test'",
        radioButtons("ttype", "Tipo de t-test:",
                     choices = list("Muestras independientes" = "indep",
                                    "Muestras relacionadas (pareadas)" = "paired"),
                     selected = "indep")
      ),
      hr(),
      downloadButton("downloadStats", "Descargar Estadísticas (Excel)"),
      br(), br(),
      downloadButton("downloadReport", "Descargar Reporte (PDF)")
    ),
    
    mainPanel(
      tabsetPanel(
        tabPanel("Inicio", 
          h4("Bienvenido a la Aplicación de Estadística"),
          p("Esta aplicación está diseñada para enseñar y aplicar análisis estadísticos con t-test y ANOVA de manera interactiva y didáctica. Aquí aprenderás tanto la teoría como la interpretación de los resultados."),
          h5("Teoría de t-test"),
          p("La prueba t permite comparar las medias de dos grupos. Se utiliza cuando se quiere determinar si existe una diferencia significativa entre ellos."),
          tags$ul(
            tags$li("Muestras independientes: Se comparan dos grupos sin relación (por ejemplo, grupo A vs grupo B)."),
            tags$li("Muestras relacionadas (pareadas): Se comparan datos emparejados (por ejemplo, antes y después de un tratamiento).")
          ),
          p("Ejemplo: Comparar la eficacia de dos medicamentos en grupos diferentes o el rendimiento de un mismo grupo antes y después de un tratamiento."),
          h5("Teoría de ANOVA"),
          p("ANOVA (Análisis de Varianza) se utiliza para comparar las medias de tres o más grupos, determinando si al menos uno difiere significativamente de los demás."),
          p("Ejemplo: Evaluar la efectividad de tres métodos de enseñanza diferentes en el rendimiento académico de los estudiantes."),
          br(),
          strong("Para comenzar, sube un archivo, selecciona las variables numéricas y haz clic en 'Analizar'.")
        ),
        
        tabPanel("Vista de Datos", 
          DTOutput("table")
        ),
        
        tabPanel("Estadísticas Descriptivas",
          h4("Medidas Descriptivas"),
          tableOutput("descriptivas"),
          br(),
          h4("Boxplot de la primera variable"),
          plotOutput("boxplot"),
          uiOutput("interp_boxplot"),
          br(),
          h4("Histograma de la primera variable"),
          plotOutput("histograma"),
          uiOutput("interp_histograma")
        ),
        
        tabPanel("Prueba Estadística",
          h4("Resultado de la Prueba"),
          verbatimTextOutput("resultado"),
          br(),
          h4("Interpretación de la Prueba"),
          verbatimTextOutput("interpretacion"),
          br(),
          h4("Gráfico Inferencial"),
          plotOutput("grafico_inferencial"),
          uiOutput("interp_inferencia")
        ),
        
        tabPanel("Detección de Datos Faltantes/Atípicos",
          h4("Validaciones"),
          verbatimTextOutput("validaciones")
        )
      )
    )
  )
)

# Servidor
server <- function(input, output, session) {
  
  # Función para calcular estadísticas descriptivas y generar un data frame adecuado
  estadisticas <- reactive({
    req(input$vars)
    df <- datos()[, input$vars, drop = FALSE]
    out <- data.frame(
      Variable = names(df),
      Media = NA,
      Mediana = NA,
      Moda = NA,
      Minimo = NA,
      Maximo = NA,
      Rango = NA,
      Desv_Est = NA,
      Coef_Var = NA,
      stringsAsFactors = FALSE
    )
    for(i in seq_along(df)){
      col <- df[[i]]
      mode_val <- Mode(col)
      if(length(mode_val) > 1) mode_val <- paste(mode_val, collapse = ",")
      out$Media[i]    <- round(mean(col, na.rm = TRUE), 3)
      out$Mediana[i]  <- round(median(col, na.rm = TRUE), 3)
      out$Moda[i]     <- mode_val
      out$Minimo[i]   <- round(min(col, na.rm = TRUE), 3)
      out$Maximo[i]   <- round(max(col, na.rm = TRUE), 3)
      out$Rango[i]    <- round(diff(range(col, na.rm = TRUE)), 3)
      out$Desv_Est[i] <- round(sd(col, na.rm = TRUE), 3)
      out$Coef_Var[i] <- round(sd(col, na.rm = TRUE)/mean(col, na.rm = TRUE), 3)
    }
    return(out)
  })
  
  # Cargar datos
  datos <- reactive({
    req(input$file)
    ext <- tools::file_ext(input$file$name)
    if (ext == "csv") {
      read.csv(input$file$datapath)
    } else if (ext == "xlsx") {
      read_excel(input$file$datapath)
    } else {
      showNotification("Formato no soportado", type = "error")
      return(NULL)
    }
  })
  
  # Vista interactiva del archivo
  output$table <- renderDT({
    req(datos())
    datatable(datos(), options = list(pageLength = 5))
  })
  
  # Seleccionar variables numéricas
  output$varSelect <- renderUI({
    req(datos())
    num_vars <- names(dplyr::select_if(datos(), is.numeric))
    tagList(
      selectInput("vars", "Selecciona 2 o 3 variables numéricas:",
                  choices = num_vars, multiple = TRUE),
      # Este input se actualizará según el análisis
      selectInput("testType", "Prueba sugerida:", choices = c("t_test", "anova"), selected = "t_test")
    )
  })
  
  # Mostrar tabla de estadísticas descriptivas
  output$descriptivas <- renderTable({
    req(estadisticas())
    estadisticas()
  })
  
  # Boxplot y su interpretación en HTML
  output$boxplot <- renderPlot({
    req(input$vars)
    ggplot(datos(), aes_string(x = "''", y = input$vars[1])) +
      geom_boxplot(fill = "lightblue") +
      labs(title = paste("Boxplot de", input$vars[1]),
           x = "", y = input$vars[1])
  })
  output$interp_boxplot <- renderUI({
    HTML("<b>Boxplot:</b><br> Muestra la mediana, los cuartiles y los posibles valores atípicos de la variable. <br>
          Los puntos fuera de los 'bigotes' indican valores atípicos.")
  })
  
  # Histograma y su interpretación en HTML
  output$histograma <- renderPlot({
    req(input$vars)
    ggplot(datos(), aes_string(x = input$vars[1])) +
      geom_histogram(binwidth = 1, fill = "skyblue", color = "black") +
      labs(title = paste("Histograma de", input$vars[1]),
           x = input$vars[1], y = "Frecuencia")
  })
  output$interp_histograma <- renderUI({
    HTML("<b>Histograma:</b><br> Representa la distribución de la variable, permitiendo observar su forma, dispersión y asimetría. <br>
          Se analizan la frecuencia y concentración de los valores.")
  })
  
  # Análisis estadístico (se ejecuta al presionar 'Analizar')
  analisis <- eventReactive(input$analyze, {
    req(input$vars)
    df <- datos()
    n <- nrow(df)
    vars <- input$vars
    # Validar selección de 2 o 3 variables
    if (!(length(vars) %in% c(2, 3))) {
      return(list(error = "Selecciona 2 variables para t-test o 3 para ANOVA."))
    }
    if (length(vars) == 2 && n > 35) {
      return(list(error = "Para t-test se recomienda ≤35 observaciones. Considera otra prueba o reduce la muestra."))
    }
    # Realizar análisis según el número de variables
    if (length(vars) == 2) {
      updateSelectInput(session, "testType", selected = "t_test")
      res <- if (input$ttype == "indep") {
        t.test(df[[vars[1]]], df[[vars[2]]])
      } else {
        t.test(df[[vars[1]]], df[[vars[2]]], paired = TRUE)
      }
      grafInfer <- ggplot(df, aes_string(x = vars[1], y = vars[2])) +
        geom_point(color = "blue") +
        labs(title = "Gráfico de Dispersión entre las variables",
             x = vars[1], y = vars[2])
      interp <- if (res$p.value < 0.05) {
        paste("Interpretación: Con un valor p =", round(res$p.value, 4),
              "se rechaza la hipótesis nula, indicando diferencias significativas entre los grupos.")
      } else {
        paste("Interpretación: Con un valor p =", round(res$p.value, 4),
              "no se rechaza la hipótesis nula, sin diferencia estadísticamente significativa.")
      }
      return(list(test = res, grafico = grafInfer, interp = interp))
    } else if (length(vars) == 3) {
      updateSelectInput(session, "testType", selected = "anova")
      formula <- as.formula(paste(vars[1], "~", paste(vars[-1], collapse = " + ")))
      modelo <- aov(formula, data = df)
      grafInfer <- ggplot(df, aes_string(x = vars[2], y = vars[1])) +
        geom_boxplot(fill = "lightgreen") +
        labs(title = paste("Boxplot de", vars[1], "por", vars[2]),
             x = vars[2], y = vars[1])
      pval <- summary(modelo)[[1]][["Pr(>F)"]][1]
      interp <- if (pval < 0.05) {
        paste("Interpretación: Con un valor p =", round(pval, 4),
              "se rechaza la hipótesis nula, indicando diferencias significativas entre los grupos.")
      } else {
        paste("Interpretación: Con un valor p =", round(pval, 4),
              "no se rechaza la hipótesis nula, sin diferencias significativas entre los grupos.")
      }
      return(list(test = summary(modelo), grafico = grafInfer, interp = interp))
    }
  })
  
  # Mostrar resultado del análisis
  output$resultado <- renderPrint({
    res <- analisis()
    if (!is.null(res$error)) {
      cat(res$error)
    } else {
      print(res$test)
    }
  })
  
  # Interpretación del análisis
  output$interpretacion <- renderPrint({
    res <- analisis()
    if (!is.null(res$error)) {
      cat(res$error)
    } else {
      cat(res$interp)
    }
  })
  
  # Gráfico inferencial y su interpretación
  output$grafico_inferencial <- renderPlot({
    req(analisis())
    if (is.null(analisis()$grafico)) return()
    analisis()$grafico
  })
  output$interp_inferencia <- renderUI({
    HTML("<b>Gráfico Inferencial:</b><br> Este gráfico muestra la relación entre las variables analizadas. <br>
          En el t-test se visualiza un gráfico de dispersión, y en ANOVA se muestra un boxplot por grupos, lo que respalda la interpretación de la prueba.")
  })
  
  # Detección de datos faltantes y outliers
  output$validaciones <- renderPrint({
    req(datos())
    df <- datos()
    faltantes <- sum(is.na(df))
    cat("Cantidad de datos faltantes:", faltantes, "\n")
    if (!is.null(input$vars) && length(input$vars) > 0) {
      for (var in input$vars) {
        qnt <- quantile(df[[var]], probs = c(0.25, 0.75), na.rm = TRUE)
        iqr <- IQR(df[[var]], na.rm = TRUE)
        outliers <- sum(df[[var]] < (qnt[1] - 1.5 * iqr) | df[[var]] > (qnt[2] + 1.5 * iqr), na.rm = TRUE)
        cat("Variable", var, ":", outliers, "valores atípicos\n")
      }
    }
  })
  
  # Descargar estadísticas a Excel
  output$downloadStats <- downloadHandler(
    filename = function() { "estadisticas.xlsx" },
    content = function(file) {
      stats <- estadisticas()
      write.xlsx(stats, file)
    }
  )
  
  # Descargar reporte en PDF mediante una plantilla R Markdown
  output$downloadReport <- downloadHandler(
    filename = function() { "reporte.pdf" },
    content = function(file) {
      tempReport <- tempfile(fileext = ".Rmd")
      # Se espera que exista un archivo 'reporte_template.Rmd' en el directorio del proyecto.
      file.copy("reporte_template.Rmd", tempReport, overwrite = TRUE)
      params <- list(data = datos(), stats = estadisticas(), analisis = analisis())
      rmarkdown::render(tempReport, output_file = file,
                        params = params,
                        envir = new.env(parent = globalenv()))
    }
  )
}

# Ejecutar la aplicación
shinyApp(ui, server)
