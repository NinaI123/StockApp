import { Link, NavLink, useNavigate } from 'react-router-dom'
import { useAuthStore } from '../store/authStore'
import { TrendingUp, LayoutDashboard, Trophy, BookOpen, User, LogOut, Swords } from 'lucide-react'

export default function Navbar() {
  const { isAuthenticated, user, logout } = useAuthStore()
  const navigate = useNavigate()

  const handleLogout = () => {
    logout()
    navigate('/')
  }

  return (
    <nav style={{
      position: 'sticky', top: 0, zIndex: 100,
      background: 'rgba(10, 14, 26, 0.92)',
      backdropFilter: 'blur(12px)',
      borderBottom: '1px solid var(--border)',
      padding: '0 24px',
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      height: '60px',
    }}>
      {/* Logo */}
      <Link to="/" style={{ display: 'flex', alignItems: 'center', gap: '8px', textDecoration: 'none' }}>
        <TrendingUp size={22} color="var(--teal)" />
        <span style={{ fontWeight: 700, fontSize: '16px', color: 'var(--text-primary)' }}>
          Fantasy<span style={{ color: 'var(--teal)' }}>Finance</span>
        </span>
      </Link>

      {/* Nav links */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
        {[
          { to: '/analyze',     icon: <TrendingUp size={15} />,      label: 'Analyze'    },
          { to: '/predictions', icon: <BookOpen size={15} />,         label: 'Predictions'},
          ...(isAuthenticated ? [
            { to: '/dashboard', icon: <LayoutDashboard size={15} />, label: 'Dashboard'  },
            { to: '/leagues',   icon: <Swords size={15} />,          label: 'Leagues'    },
            { to: '/portfolio', icon: <BookOpen size={15} />,         label: 'Portfolio'  },
          ] : []),
        ].map(({ to, icon, label }) => (
          <NavLink key={to} to={to} style={({ isActive }) => ({
            display: 'flex', alignItems: 'center', gap: '5px',
            padding: '6px 12px', borderRadius: '6px', textDecoration: 'none',
            fontSize: '13px', fontWeight: 500,
            color: isActive ? 'var(--teal)' : 'var(--text-secondary)',
            background: isActive ? 'var(--teal-dim)' : 'transparent',
            transition: 'all 0.15s',
          })}>
            {icon} {label}
          </NavLink>
        ))}
      </div>

      {/* Auth */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
        {isAuthenticated ? (
          <>
            <NavLink to={`/profile/${user?.user_id}`} style={{ textDecoration: 'none' }}>
              <div style={{
                display: 'flex', alignItems: 'center', gap: '8px',
                padding: '5px 10px', borderRadius: '8px', border: '1px solid var(--border)',
                color: 'var(--text-secondary)', fontSize: '13px', cursor: 'pointer',
                transition: 'all 0.15s',
              }}>
                <User size={14} />
                <span>{user?.email?.split('@')[0]}</span>
              </div>
            </NavLink>
            <button onClick={handleLogout} style={{
              display: 'flex', alignItems: 'center', gap: '5px',
              padding: '6px 12px', borderRadius: '6px', border: 'none',
              background: 'transparent', color: 'var(--text-muted)', cursor: 'pointer',
              fontSize: '13px', transition: 'color 0.15s',
            }}
            onMouseEnter={e => e.target.style.color='var(--coral)'}
            onMouseLeave={e => e.target.style.color='var(--text-muted)'}
            >
              <LogOut size={14} /> Logout
            </button>
          </>
        ) : (
          <>
            <Link to="/login" style={{
              padding: '7px 14px', borderRadius: '7px', textDecoration: 'none',
              color: 'var(--text-secondary)', fontSize: '13px', fontWeight: 500,
              border: '1px solid var(--border)', transition: 'all 0.15s',
            }}>Log in</Link>
            <Link to="/signup" style={{
              padding: '7px 16px', borderRadius: '7px', textDecoration: 'none',
              background: 'var(--teal)', color: '#fff', fontSize: '13px', fontWeight: 600,
              transition: 'all 0.15s',
            }}>Get started</Link>
          </>
        )}
      </div>
    </nav>
  )
}
