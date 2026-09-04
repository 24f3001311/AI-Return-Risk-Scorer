<template>
  <header class="app-header">
    <div class="logo">
      <div class="logo-icon">[+]</div>
      <div class="logo-text">AI Return-Risk <span>Scorer</span></div>
    </div>
    
    <div class="status-badge" :class="isBackendOnline ? 'online' : 'offline'">
      <div class="status-dot" :class="isBackendOnline ? 'online' : 'offline'"></div>
      {{ isBackendOnline ? 'ML Engine Online' : 'Connecting...' }}
    </div>
  </header>

  <main class="main-layout">
    <!-- Left Column: Input -->
    <JsonInput 
      @evaluate="handleEvaluate"
      :loading="isEvaluating"
    />
    
    <!-- Right Column: Results -->
    <div class="results-panel">
      <RiskGauge 
        v-if="currentResponse"
        :score="currentResponse.risk_score"
        :action="currentResponse.action"
        :routing="currentResponse.routing"
      />
      <div v-else class="risk-gauge-container glass-card" style="display: flex; align-items: center; justify-content: center; min-height: 250px;">
        <div class="empty-state" style="padding: 0;">
          <div class="icon" style="font-size: 2rem;">[ ]</div>
          <p>Ready to evaluate transaction risk.</p>
        </div>
      </div>
      
      <ShapWaterfall 
        :reasons="currentResponse?.reasons || []"
      />
    </div>
    
    <!-- Full Width: History -->
    <HistoryTable :history="evaluationHistory" />
  </main>
</template>

<script>
import api from './services/api'
import JsonInput from './components/JsonInput.vue'
import RiskGauge from './components/RiskGauge.vue'
import ShapWaterfall from './components/ShapWaterfall.vue'
import HistoryTable from './components/HistoryTable.vue'

export default {
  name: 'App',
  components: {
    JsonInput,
    RiskGauge,
    ShapWaterfall,
    HistoryTable
  },
  data() {
    return {
      isBackendOnline: false,
      isEvaluating: false,
      currentResponse: null,
      evaluationHistory: [],
      healthCheckInterval: null
    }
  },
  methods: {
    async checkBackendHealth() {
      try {
        const health = await api.checkHealth()
        this.isBackendOnline = health.status === 'healthy' && health.model_loaded
      } catch (e) {
        this.isBackendOnline = false
      }
    },
    
    async handleEvaluate(payload) {
      if (this.isEvaluating) return
      
      this.isEvaluating = true
      try {
        const response = await api.evaluateTransaction(payload)
        this.currentResponse = response
        
        // Add to history
        this.evaluationHistory.unshift({
          id: Date.now(),
          timestamp: new Date().toISOString(),
          payload: payload,
          response: response
        })
        
        // Keep only last 10
        if (this.evaluationHistory.length > 10) {
          this.evaluationHistory.pop()
        }
      } catch (error) {
        console.error('Evaluation failed:', error)
        alert(error.response?.data?.detail || error.message || 'Evaluation failed')
      } finally {
        this.isEvaluating = false
      }
    }
  },
  mounted() {
    // Initial health check
    this.checkBackendHealth()
    
    // Poll health every 10 seconds
    this.healthCheckInterval = setInterval(this.checkBackendHealth, 10000)
  },
  beforeUnmount() {
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval)
    }
  }
}
</script>
