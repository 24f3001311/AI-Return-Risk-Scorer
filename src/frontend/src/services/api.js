/**
 * API Service — Axios wrapper for communicating with the FastAPI backend.
 */

import axios from 'axios'

const API_BASE_URL = '/api'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000,
})

/**
 * Evaluate a transaction payload for return-risk.
 * @param {Object} payload - Transaction JSON payload
 * @returns {Promise<Object>} Risk evaluation response
 */
export async function evaluateTransaction(payload) {
  const response = await apiClient.post('/evaluate', payload)
  return response.data
}

/**
 * Check API health status.
 * @returns {Promise<Object>} Health status
 */
export async function checkHealth() {
  const response = await apiClient.get('/health')
  return response.data
}

export default { evaluateTransaction, checkHealth }
