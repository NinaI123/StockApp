import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { useAuthStore } from './store/authStore'

import Landing       from './pages/Landing'
import Login         from './pages/Login'
import Signup        from './pages/Signup'
import StockAnalysis from './pages/StockAnalysis'
import Dashboard     from './pages/Dashboard'
import LeaguesHub    from './pages/LeaguesHub'
import LeagueDetail  from './pages/LeagueDetail'
import DraftRoom     from './pages/DraftRoom'
import MatchupDetail from './pages/MatchupDetail'
import Predictions   from './pages/Predictions'
import Portfolio     from './pages/Portfolio'
import Profile       from './pages/Profile'
import Navbar        from './components/Navbar'

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: 1, staleTime: 30_000 } }
})

function PrivateRoute({ children }) {
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated)
  return isAuthenticated ? children : <Navigate to="/login" replace />
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Navbar />
        <Routes>
          <Route path="/"            element={<Landing />} />
          <Route path="/login"       element={<Login />} />
          <Route path="/signup"      element={<Signup />} />
          <Route path="/analyze"     element={<StockAnalysis />} />
          <Route path="/dashboard"   element={<PrivateRoute><Dashboard /></PrivateRoute>} />
          <Route path="/leagues"     element={<PrivateRoute><LeaguesHub /></PrivateRoute>} />
          <Route path="/leagues/:id" element={<PrivateRoute><LeagueDetail /></PrivateRoute>} />
          <Route path="/leagues/:id/draft" element={<PrivateRoute><DraftRoom /></PrivateRoute>} />
          <Route path="/matchups/:id"    element={<PrivateRoute><MatchupDetail /></PrivateRoute>} />
          <Route path="/predictions" element={<Predictions />} />
          <Route path="/portfolio"   element={<PrivateRoute><Portfolio /></PrivateRoute>} />
          <Route path="/profile/:id" element={<Profile />} />
          <Route path="*"            element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
