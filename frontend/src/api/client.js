import axios from 'axios'
import { useAuthStore } from '../store/authStore'

const client = axios.create({
  baseURL: '/api',
  headers: { 'Content-Type': 'application/json' },
})

// Attach JWT on every request
client.interceptors.request.use((config) => {
  const token = useAuthStore.getState().token
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

// Auto-logout on 401
client.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      useAuthStore.getState().logout()
    }
    return Promise.reject(err)
  }
)

export default client

// ── Stocks ──────────────────────────────────────────────────────────────────
export const fetchAnalysis   = (symbol) => client.get(`/stock/analysis?symbol=${symbol}`).then(r => r.data)
export const fetchSummary    = (symbol) => client.get(`/stock/summary?symbol=${symbol}`).then(r => r.data)
export const fetchHistorical = (symbol, period = '1y') => client.get(`/stock/historical?symbol=${symbol}&period=${period}`).then(r => r.data)
export const fetchNews       = (symbol) => client.get(`/news?symbol=${symbol}`).then(r => r.data)
export const fetchInsights   = () => client.get('/insights/daily').then(r => r.data)

// ── Auth ─────────────────────────────────────────────────────────────────────
export const signup = (email, password) => client.post('/auth/signup', { email, password }).then(r => r.data)
export const login  = (email, password) => client.post('/auth/login',  { email, password }).then(r => r.data)

// ── Portfolio ─────────────────────────────────────────────────────────────────
export const fetchTrades    = () => client.get('/portfolio/trades').then(r => r.data)
export const addTrade       = (trade) => client.post('/portfolio/trades', trade).then(r => r.data)
export const fetchWatchlist = () => client.get('/portfolio/watchlist').then(r => r.data)
export const addWatch       = (symbol) => client.post('/portfolio/watchlist', { symbol }).then(r => r.data)
export const removeWatch    = (symbol) => client.delete('/portfolio/watchlist', { data: { symbol } }).then(r => r.data)

// ── Wars ──────────────────────────────────────────────────────────────────────
export const createLeague      = (body) => client.post('/wars/league', body).then(r => r.data)
export const joinLeague        = (body) => client.post('/wars/team', body).then(r => r.data)
export const fetchLeague       = (ref)  => client.get(`/wars/league/${ref}`).then(r => r.data)
export const fetchPublicLeagues= (page=1) => client.get(`/wars/leagues/public?page=${page}`).then(r => r.data)
export const createMatchup     = (body) => client.post('/wars/matchup', body).then(r => r.data)
export const scoreMatchup      = (id)   => client.post(`/wars/score/${id}`, {}).then(r => r.data)
export const fetchReport       = (id)   => client.get(`/wars/matchups/${id}/report`).then(r => r.data)
export const submitDraftPick   = (body) => client.post('/wars/draft', body).then(r => r.data)
export const postMessage       = (body) => client.post('/wars/message', body).then(r => r.data)
export const fetchMessages     = (lid)  => client.get(`/wars/messages/${lid}`).then(r => r.data)
export const likeMessage       = (mid)  => client.post(`/wars/message/${mid}/like`).then(r => r.data)

// ── Predictions ────────────────────────────────────────────────────────────────
export const fetchPredictions  = (page=1) => client.get(`/predictions/feed?page=${page}`).then(r => r.data)
export const createPrediction  = (body)  => client.post('/predictions', body).then(r => r.data)
export const likePrediction    = (id)    => client.post(`/predictions/${id}/like`).then(r => r.data)

// ── Profile ────────────────────────────────────────────────────────────────────
export const fetchProfile = (id) => client.get(`/users/${id}/profile`).then(r => r.data)
