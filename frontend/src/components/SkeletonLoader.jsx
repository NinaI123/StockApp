function Skeleton({ width = '100%', height = '16px', style = {} }) {
  return <div className="skeleton" style={{ width, height, ...style }} />
}

export function SkeletonCard() {
  return (
    <div className="card fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
      <Skeleton height="12px" width="60px" />
      <Skeleton height="28px" width="140px" />
      <Skeleton height="14px" width="90%" />
      <Skeleton height="14px" width="75%" />
      <Skeleton height="14px" width="80%" />
    </div>
  )
}

export function SkeletonSignalCard() {
  return (
    <div className="card fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <Skeleton height="12px" width="80px" />
          <Skeleton height="32px" width="120px" />
        </div>
        <Skeleton height="36px" width="100px" style={{ borderRadius: '20px' }} />
      </div>
      <Skeleton height="1px" />
      {[...Array(5)].map((_, i) => (
        <div key={i} style={{ display: 'flex', justifyContent: 'space-between' }}>
          <Skeleton height="12px" width="100px" />
          <Skeleton height="12px" width="80px" />
        </div>
      ))}
    </div>
  )
}

export default Skeleton
