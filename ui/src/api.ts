import type { Analysis, Match } from './types'

type ProgressCb = (step: number, total: number, message: string) => void
type AnalysisCb = (analysis: Analysis) => void
type MatchesCb  = (matches: Match[]) => void
type ErrorCb    = (message: string) => void

export async function analyzeImage(
  file: File,
  onProgress: ProgressCb,
  onAnalysis: AnalysisCb,
  onMatches:  MatchesCb,
  onError:    ErrorCb,
): Promise<void> {
  const form = new FormData()
  form.append('file', file)

  const res = await fetch('/api/analyze', { method: 'POST', body: form })
  if (!res.ok || !res.body) {
    onError(`HTTP ${res.status}: ${res.statusText}`)
    return
  }

  const reader  = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer    = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    // Parse SSE events from buffer
    const events = buffer.split('\n\n')
    buffer = events.pop() ?? ''

    for (const block of events) {
      const lines = block.trim().split('\n')
      let event = ''
      let data  = ''
      for (const line of lines) {
        if (line.startsWith('event: ')) event = line.slice(7)
        if (line.startsWith('data: '))  data  = line.slice(6)
      }
      if (!event || !data) continue

      try {
        const payload = JSON.parse(data)
        if (event === 'progress') onProgress(payload.step, payload.total, payload.message)
        if (event === 'analysis') onAnalysis(payload as Analysis)
        if (event === 'matches')  onMatches(payload.matches as Match[])
        if (event === 'error')    onError(payload.message)
      } catch {
        // malformed SSE chunk — skip
      }
    }
  }
}
