// JavaScript application logic for Sardonix AI Sarcasm Detector

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const tabs = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');
  const analyzeBtn = document.getElementById('analyze-btn');
  const textInput = document.getElementById('text-input');
  const btnText = document.getElementById('btn-text');
  const btnSpinner = document.getElementById('btn-spinner');
  
  const resultsWrapper = document.getElementById('results-wrapper');
  const explainSection = document.getElementById('explain-section');
  const gaugeFill = document.getElementById('gauge-fill');
  const gaugeVal = document.getElementById('gauge-val');
  const verdictBadge = document.getElementById('verdict-badge');
  const verdictDesc = document.getElementById('verdict-desc');
  const tokenMap = document.getElementById('token-map');
  const presetChips = document.querySelectorAll('.preset-chip');
  const modelSelect = document.getElementById('model-select');

  // Metrics DOM Elements
  const metricAcc = document.getElementById('metric-acc');
  const metricPrec = document.getElementById('metric-prec');
  const metricRec = document.getElementById('metric-rec');
  const metricF1 = document.getElementById('metric-f1');
  
  const cmTP = document.getElementById('cm-tp');
  const cmFP = document.getElementById('cm-fp');
  const cmTN = document.getElementById('cm-tn');
  const cmFN = document.getElementById('cm-fn');
  
  const sarcasticVocabList = document.getElementById('sarcastic-vocab-list');
  const literalVocabList = document.getElementById('literal-vocab-list');

  // Dataset Explorer DOM Elements
  const explorerSearch = document.getElementById('explorer-search');
  const explorerSplit = document.getElementById('explorer-split');
  const explorerLabel = document.getElementById('explorer-label');
  const tweetList = document.getElementById('tweet-list');
  const explorerPrevBtn = document.getElementById('explorer-prev-btn');
  const explorerNextBtn = document.getElementById('explorer-next-btn');
  const explorerPageInfo = document.getElementById('explorer-page-info');

  // State Variables
  let metricsData = null;
  let chartInstance = null;
  let selectedModel = 'lr';
  
  // Explorer State
  let explorerQuery = '';
  let explorerSplitVal = 'train';
  let explorerLabelVal = 'all';
  let explorerOffset = 0;
  const explorerLimit = 10;
  let explorerTotal = 0;

  // Initialize
  fetchMetrics();
  setupTabs();
  setupPresets();
  setupModelSelect();
  setupExplorer();

  // Tab switching setup
  function setupTabs() {
    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        const target = tab.dataset.tab;
        
        tabs.forEach(t => t.classList.remove('active'));
        tabContents.forEach(c => c.classList.remove('active'));
        
        tab.classList.add('active');
        document.getElementById(`${target}-tab`).classList.add('active');
        
        // If Explorer tab is activated and we haven't loaded tweets yet, fetch them
        if (target === 'explorer' && explorerTotal === 0) {
          fetchTweets();
        }
      });
    });
  }

  // Preset chips setup
  function setupPresets() {
    presetChips.forEach(chip => {
      chip.addEventListener('click', () => {
        textInput.value = chip.dataset.text;
        analyzeText();
      });
    });
  }

  // Model Selector setup
  function setupModelSelect() {
    modelSelect.addEventListener('change', (e) => {
      selectedModel = e.target.value;
      
      // Update the performance metrics tab
      updateMetricsUI();
      
      // If there is currently a text in the predictor, run the analysis using the new model!
      if (textInput.value.trim().length > 0) {
        analyzeText();
      }
    });
  }

  // Fetch metrics on load
  async function fetchMetrics() {
    try {
      const response = await fetch('/api/metrics');
      if (!response.ok) throw new Error('Failed to fetch metrics');
      metricsData = await response.json();
      
      // Update UI with metrics
      updateMetricsUI();
    } catch (error) {
      console.error('Error fetching metrics:', error);
    }
  }

  function updateMetricsUI() {
    if (!metricsData) return;
    
    // Extract metrics for the selected model
    const currentModelMetrics = metricsData.models[selectedModel];
    if (!currentModelMetrics) return;

    // Grid metrics
    metricAcc.textContent = (currentModelMetrics.accuracy * 100).toFixed(1) + '%';
    metricPrec.textContent = (currentModelMetrics.precision * 100).toFixed(1) + '%';
    metricRec.textContent = (currentModelMetrics.recall * 100).toFixed(1) + '%';
    metricF1.textContent = (currentModelMetrics.f1_score * 100).toFixed(1) + '%';

    // Confusion Matrix (Format: [[TN, FP], [FN, TP]])
    const cm = currentModelMetrics.confusion_matrix;
    cmTN.textContent = cm[0][0];
    cmFP.textContent = cm[0][1];
    cmFN.textContent = cm[1][0];
    cmTP.textContent = cm[1][1];

    // Populate vocab lists
    sarcasticVocabList.innerHTML = '';
    currentModelMetrics.top_sarcastic.slice(0, 15).forEach(item => {
      sarcasticVocabList.appendChild(createVocabItemRow(item.token, item.coef, true));
    });

    literalVocabList.innerHTML = '';
    currentModelMetrics.top_literal.slice(0, 15).forEach(item => {
      literalVocabList.appendChild(createVocabItemRow(item.token, item.coef, false));
    });

    // Render features chart
    renderFeaturesChart();
  }

  function createVocabItemRow(token, coef, isSarcastic) {
    const row = document.createElement('div');
    row.className = 'vocab-item';
    
    const tokenSpan = document.createElement('span');
    tokenSpan.className = 'vocab-token';
    tokenSpan.textContent = token;
    
    const coefSpan = document.createElement('span');
    coefSpan.className = `vocab-coef ${isSarcastic ? 'pos' : 'neg'}`;
    coefSpan.textContent = (coef > 0 ? '+' : '') + coef.toFixed(3);
    
    row.appendChild(tokenSpan);
    row.appendChild(coefSpan);
    return row;
  }

  // Analyze text event listener
  analyzeBtn.addEventListener('click', () => analyzeText());

  async function analyzeText() {
    const text = textInput.value.trim();
    if (!text) return;

    // UI Loading State
    analyzeBtn.disabled = true;
    btnSpinner.style.display = 'block';
    btnText.textContent = 'Analyzing...';
    
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, model: selectedModel })
      });

      if (!response.ok) throw new Error('API server returned error');
      const result = await response.json();

      renderPredictionResult(result);
    } catch (error) {
      console.error('Error analyzing text:', error);
      alert('An error occurred while connecting to the Sardonix NLP backend server.');
    } finally {
      // Restore Button State
      analyzeBtn.disabled = false;
      btnSpinner.style.display = 'none';
      btnText.textContent = 'Analyze Text';
    }
  }

  function renderPredictionResult(result) {
    // Show sections
    resultsWrapper.classList.add('active');
    explainSection.classList.add('active');

    // Update gauge
    const prob = result.probability;
    const offset = 251.2 * (1 - prob);
    gaugeFill.style.strokeDashoffset = offset;
    
    // Choose color for gauge fill
    if (result.prediction === 1) {
      gaugeFill.style.stroke = '#ffffff';
      gaugeFill.style.filter = 'none';
    } else {
      gaugeFill.style.stroke = '#444444';
      gaugeFill.style.filter = 'none';
    }
    
    gaugeVal.textContent = Math.round(prob * 100) + '%';

    // Update Verdict Badge
    verdictBadge.className = 'verdict-badge';
    if (result.prediction === 1) {
      verdictBadge.classList.add('sarcastic');
      verdictBadge.innerHTML = `
        <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm3.5-9c.83 0 1.5-.67 1.5-1.5S16.33 8 15.5 8 14 8.67 14 9.5s.67 1.5 1.5 1.5zm-7 0c.83 0 1.5-.67 1.5-1.5S9.33 8 8.5 8 7 8.67 7 9.5 7.67 11 8.5 11zm1.79 3.25c.57-.62 1.4-.95 2.21-.95.82 0 1.64.33 2.21.95.14.15.36.17.52.05.18-.13.22-.38.09-.56-.69-.76-1.72-1.19-2.82-1.19-1.1 0-2.13.43-2.82 1.19-.13.18-.09.43.09.56.16.12.38.1.52-.05z"/></svg>
        Sarcastic
      `;
      verdictDesc.textContent = `The model detected sarcasm with a confidence score of ${(prob * 100).toFixed(1)}%. Key sarcastic keywords heavily influenced this outcome.`;
    } else {
      verdictBadge.classList.add('literal');
      verdictBadge.innerHTML = `
        <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M11.99 2C6.47 2 2 6.48 2 12s4.47 10 9.99 10C17.52 22 22 17.52 22 12S17.52 2 11.99 2zm6.93 6h-2.95a15.65 15.65 0 0 0-1.38-3.56A8.03 8.03 0 0 1 18.92 8zM12 4.04c.83 1.2 1.48 2.53 1.91 3.96h-3.82c.43-1.43 1.08-2.76 1.91-3.96zM4.26 14C4.1 13.35 4 12.68 4 12s.1-1.35.26-2h3.38c-.08.66-.14 1.33-.14 2s.06 1.34.14 2H4.26zm.82 2h2.95c.32 1.25.78 2.45 1.38 3.56A7.987 7.987 0 0 1 5.08 16zm2.95-8H5.08a7.987 7.987 0 0 1 4.33-3.56A15.65 15.65 0 0 0 8.03 8zM12 19.96c-.83-1.2-1.48-2.53-1.91-3.96h3.82c-.43 1.43-1.08 2.76-1.91 3.96zM14.34 14H9.66c-.09-.66-.16-1.34-.16-2s.07-1.34.16-2h4.68c.09.66.16 1.34.16 2s-.07 1.34-.16 2zm.25 5.56c.6-1.11 1.06-2.31 1.38-3.56h2.95a7.987 7.987 0 0 1-4.33 3.56zM16.36 14c.08-.66.14-1.33.14-2s-.06-1.34-.14-2h3.38c.16.65.26 1.32.26 2s-.1 1.35-.26 2h-3.38z"/></svg>
        Literal
      `;
      verdictDesc.textContent = `The model classified this text as literal/non-ironic. Sarcasm confidence is low at ${(prob * 100).toFixed(1)}%.`;
    }

    // Render explainability token map
    tokenMap.innerHTML = '';
    const tokens = result.tokens;
    
    // Find maximum absolute contribution to normalize opacity scales
    const maxContrib = Math.max(...tokens.map(t => Math.abs(t.contribution)), 0.0001);
    
    tokens.forEach(tok => {
      const span = document.createElement('span');
      span.className = 'token-span';
      span.textContent = tok.token;
      
      const contrib = tok.contribution;
      const absContrib = Math.abs(contrib);
      
      // Calculate opacity between 0.05 and 0.85
      const intensity = absContrib / maxContrib;
      const opacity = 0.05 + 0.8 * intensity;
      
      if (contrib > 0.0001) {
        // Sarcastic (White background, black text when high opacity)
        span.style.backgroundColor = `rgba(255, 255, 255, ${opacity.toFixed(2)})`;
        span.style.color = opacity > 0.4 ? '#000000' : '#ffffff';
        span.style.borderBottom = `2px solid #ffffff`;
      } else if (contrib < -0.0001) {
        // Literal (Medium Gray background)
        span.style.backgroundColor = `rgba(136, 136, 136, ${(opacity * 0.7).toFixed(2)})`;
        span.style.color = '#ffffff';
        span.style.borderBottom = `2px solid #666666`;
      } else {
        // Neutral
        span.style.borderBottom = `1px solid rgba(255, 255, 255, 0.1)`;
      }
      
      // Create detailed info tooltip text
      const tooltip = 
`Token: "${tok.token}"
Contribution: ${(contrib > 0 ? '+' : '') + contrib.toFixed(4)}
TF-IDF Value: ${tok.tfidf.toFixed(4)}
Model Coefficient: ${(tok.coefficient > 0 ? '+' : '') + tok.coefficient.toFixed(4)}`;
      
      span.setAttribute('data-tooltip', tooltip);
      tokenMap.appendChild(span);
    });
  }

  // Render Top Features Chart
  function renderFeaturesChart() {
    if (!metricsData) return;

    const currentModelMetrics = metricsData.models[selectedModel];
    if (!currentModelMetrics) return;

    const ctx = document.getElementById('featuresChart').getContext('2d');
    
    // Extract top 8 sarcastic and top 8 literal features for comparison
    const topSarcastic = currentModelMetrics.top_sarcastic.slice(0, 8);
    const topLiteral = currentModelMetrics.top_literal.slice(0, 8).reverse(); // Reverse so most negative is at the top
    
    const tokens = [...topLiteral.map(item => item.token), ...topSarcastic.map(item => item.token)];
    const values = [...topLiteral.map(item => item.coef), ...topSarcastic.map(item => item.coef)];
    
    // Color bars: gray for literal (negative), white for sarcastic (positive)
    const backgroundColors = [
      ...topLiteral.map(() => 'rgba(100, 100, 100, 0.6)'),
      ...topSarcastic.map(() => 'rgba(255, 255, 255, 0.8)')
    ];
    
    const borderColors = [
      ...topLiteral.map(() => 'rgba(120, 120, 120, 1)'),
      ...topSarcastic.map(() => 'rgba(255, 255, 255, 1)')
    ];

    if (chartInstance) {
      chartInstance.destroy();
    }

    chartInstance = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: tokens,
        datasets: [{
          label: 'Model Feature Weights (Coefficients)',
          data: values,
          backgroundColor: backgroundColors,
          borderColor: borderColors,
          borderWidth: 1.5,
          borderRadius: 4
        }]
      },
      options: {
        indexAxis: 'y', // Horizontal bars
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: false
          },
          tooltip: {
            callbacks: {
              label: function(context) {
                const val = context.raw;
                return `Weight: ${(val > 0 ? '+' : '') + val.toFixed(4)}`;
              }
            },
            backgroundColor: '#18181b',
            titleColor: '#fff',
            bodyColor: '#e0e0e0',
            borderColor: '#27272a',
            borderWidth: 1
          }
        },
        scales: {
          x: {
            grid: {
              color: 'rgba(255, 255, 255, 0.05)',
              drawBorder: false
            },
            ticks: {
              color: '#8e8e9f',
              font: { family: 'Inter' }
            }
          },
          y: {
            grid: {
              display: false
            },
            ticks: {
              color: '#ffffff',
              font: { family: 'Inter', weight: '500' }
            }
          }
        }
      }
    });
  }

  // Dataset Explorer Integration
  function setupExplorer() {
    let debounceTimer;
    
    // Keyword search event listener (with debouncing)
    explorerSearch.addEventListener('input', (e) => {
      explorerQuery = e.target.value;
      explorerOffset = 0;
      
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        fetchTweets();
      }, 350);
    });

    // Split selection change
    explorerSplit.addEventListener('change', (e) => {
      explorerSplitVal = e.target.value;
      explorerOffset = 0;
      fetchTweets();
    });

    // Label selection change
    explorerLabel.addEventListener('change', (e) => {
      explorerLabelVal = e.target.value;
      explorerOffset = 0;
      fetchTweets();
    });

    // Pagination buttons
    explorerPrevBtn.addEventListener('click', () => {
      if (explorerOffset >= explorerLimit) {
        explorerOffset -= explorerLimit;
        fetchTweets();
      }
    });

    explorerNextBtn.addEventListener('click', () => {
      if (explorerOffset + explorerLimit < explorerTotal) {
        explorerOffset += explorerLimit;
        fetchTweets();
      }
    });
  }

  // Fetch Tweets API request
  async function fetchTweets() {
    try {
      tweetList.innerHTML = '<div style="text-align: center; color: var(--text-muted); font-style: italic; padding: 2rem;">Loading tweets...</div>';
      
      const url = `/api/tweets?query=${encodeURIComponent(explorerQuery)}&split=${explorerSplitVal}&label=${explorerLabelVal}&limit=${explorerLimit}&offset=${explorerOffset}`;
      
      const response = await fetch(url);
      if (!response.ok) throw new Error('Failed to fetch tweets');
      
      const data = await response.json();
      explorerTotal = data.total;
      
      renderTweets(data.tweets);
      updatePaginationUI();
    } catch (error) {
      console.error('Error fetching tweets:', error);
      tweetList.innerHTML = '<div style="text-align: center; color: rgba(255,0,0,0.5); font-style: italic; padding: 2rem;">Error loading tweets.</div>';
    }
  }

  // Render search results
  function renderTweets(tweets) {
    tweetList.innerHTML = '';
    
    if (tweets.length === 0) {
      tweetList.innerHTML = '<div style="text-align: center; color: var(--text-muted); font-style: italic; padding: 2.5rem;">No matching tweets found.</div>';
      return;
    }

    tweets.forEach(tweet => {
      const card = document.createElement('div');
      card.className = 'tweet-card';
      
      const isSarcastic = tweet.label === 1;
      
      const header = document.createElement('div');
      header.className = 'tweet-card-header';
      
      const idSpan = document.createElement('span');
      idSpan.className = 'tweet-card-id';
      idSpan.textContent = `Tweet #${tweet.index}`;
      
      const badge = document.createElement('span');
      badge.className = `tweet-card-badge ${isSarcastic ? 'sarcastic' : 'literal'}`;
      badge.textContent = isSarcastic ? 'Sarcastic' : 'Literal';
      
      header.appendChild(idSpan);
      header.appendChild(badge);
      
      const textDiv = document.createElement('div');
      textDiv.className = 'tweet-card-text';
      textDiv.textContent = tweet.text;
      
      card.appendChild(header);
      card.appendChild(textDiv);
      
      // Click card to load into predictor and run analysis
      card.addEventListener('click', () => {
        textInput.value = tweet.text;
        
        // Focus the text area
        textInput.focus();
        
        // Perform prediction
        analyzeText();
      });
      
      tweetList.appendChild(card);
    });
  }

  // Update pagination elements
  function updatePaginationUI() {
    const from = explorerTotal === 0 ? 0 : explorerOffset + 1;
    const to = Math.min(explorerOffset + explorerLimit, explorerTotal);
    
    explorerPageInfo.textContent = `Showing ${from}-${to} of ${explorerTotal}`;
    
    explorerPrevBtn.disabled = explorerOffset === 0;
    explorerNextBtn.disabled = explorerOffset + explorerLimit >= explorerTotal;
  }
});
