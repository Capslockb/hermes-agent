import { FitAddon } from '@xterm/addon-fit'
import { Unicode11Addon } from '@xterm/addon-unicode11'
import { WebLinksAddon } from '@xterm/addon-web-links'
import { Terminal } from '@xterm/xterm'
import { useCallback, useEffect, useRef, useState } from 'react'
import type { CSSProperties } from 'react'

import { triggerHaptic } from '@/lib/haptics'
import { useTheme } from '@/themes/context'

import { isAddSelectionShortcut, terminalSelectionAnchor, terminalSelectionLabel, terminalTheme } from './selection'

type TerminalStatus = 'closed' | 'open' | 'starting'

interface UseTerminalSessionOptions {
  cwd: string
  onAddSelectionToChat: (text: string, label?: string) => void
}

export function useTerminalSession({ cwd, onAddSelectionToChat }: UseTerminalSessionOptions) {
  const hostRef = useRef<HTMLDivElement | null>(null)
  const termRef = useRef<Terminal | null>(null)
  const sessionIdRef = useRef<string | null>(null)
  const shellNameRef = useRef('shell')
  const selectionLabelRef = useRef('')
  const selectionRef = useRef('')
  const onAddSelectionToChatRef = useRef(onAddSelectionToChat)
  const { resolvedMode } = useTheme()
  const resolvedModeRef = useRef(resolvedMode)
  const [status, setStatus] = useState<TerminalStatus>('starting')
  const [selection, setSelection] = useState('')
  const [selectionStyle, setSelectionStyle] = useState<CSSProperties | null>(null)
  const [shellName, setShellName] = useState('shell')

  useEffect(() => {
    onAddSelectionToChatRef.current = onAddSelectionToChat
  }, [onAddSelectionToChat])

  useEffect(() => {
    resolvedModeRef.current = resolvedMode

    if (termRef.current) {
      termRef.current.options.theme = terminalTheme(resolvedMode)
    }
  }, [resolvedMode])

  const addSelectionToChat = useCallback(() => {
    const selectedText = selectionRef.current || termRef.current?.getSelection() || ''

    const label =
      selectionLabelRef.current ||
      (termRef.current ? terminalSelectionLabel(termRef.current, shellNameRef.current, selectedText) : 'selection')

    const trimmed = selectedText.trim()

    if (!trimmed) {
      return
    }

    onAddSelectionToChatRef.current(trimmed, label)
    termRef.current?.clearSelection()
    selectionRef.current = ''
    selectionLabelRef.current = ''
    setSelection('')
    setSelectionStyle(null)
    triggerHaptic('selection')
  }, [])

  useEffect(() => {
    if (!selection.trim()) {
      return
    }

    const onKeyDown = (event: KeyboardEvent) => {
      if (!isAddSelectionShortcut(event)) {
        return
      }

      event.preventDefault()
      event.stopPropagation()
      addSelectionToChat()
    }

    window.addEventListener('keydown', onKeyDown, { capture: true })

    return () => window.removeEventListener('keydown', onKeyDown, { capture: true })
  }, [addSelectionToChat, selection])

  useEffect(() => {
    const host = hostRef.current
    const terminalApi = window.hermesDesktop?.terminal

    if (!host || !terminalApi) {
      setStatus('closed')

      return
    }

    let disposed = false
    const cleanup: Array<() => void> = []

    const term = new Terminal({
      allowProposedApi: true,
      allowTransparency: true,
      convertEol: true,
      cursorBlink: true,
      fontFamily: "'SF Mono', 'Menlo', 'Cascadia Code', 'JetBrains Mono', monospace",
      fontSize: 11,
      lineHeight: 1.12,
      macOptionIsMeta: true,
      scrollback: 1000,
      theme: terminalTheme(resolvedModeRef.current)
    })

    const fit = new FitAddon()

    termRef.current = term
    term.loadAddon(fit)
    term.loadAddon(new Unicode11Addon())
    term.loadAddon(new WebLinksAddon())
    term.unicode.activeVersion = '11'
    term.open(host)

    const fitAndResize = () => {
      if (disposed || !host.isConnected || host.clientWidth <= 0 || host.clientHeight <= 0) {
        return
      }

      try {
        fit.fit()
      } catch {
        return
      }

      const id = sessionIdRef.current

      if (id) {
        void terminalApi.resize(id, { cols: term.cols, rows: term.rows })
      }
    }

    const resizeObserver = new ResizeObserver(fitAndResize)
    resizeObserver.observe(host)
    cleanup.push(() => resizeObserver.disconnect())

    const dataDisposable = term.onData(data => {
      const id = sessionIdRef.current

      if (id) {
        void terminalApi.write(id, data)
      }
    })

    cleanup.push(() => dataDisposable.dispose())

    const selectionDisposable = term.onSelectionChange(() => {
      const next = term.getSelection()
      selectionRef.current = next
      selectionLabelRef.current = next.trim() ? terminalSelectionLabel(term, shellNameRef.current, next) : ''
      setSelection(next)
      setSelectionStyle(next.trim() ? terminalSelectionAnchor(host) : null)
    })

    cleanup.push(() => selectionDisposable.dispose())

    term.attachCustomKeyEventHandler(event => {
      if (event.type !== 'keydown') {
        return true
      }

      if (isAddSelectionShortcut(event) && term.hasSelection()) {
        event.preventDefault()
        addSelectionToChat()

        return false
      }

      return true
    })

    fitAndResize()

    void terminalApi
      .start({ cols: term.cols, cwd, rows: term.rows })
      .then(session => {
        if (disposed) {
          void terminalApi.dispose(session.id)

          return
        }

        sessionIdRef.current = session.id
        shellNameRef.current = session.shell || 'shell'
        setShellName(session.shell || 'shell')

        if (term.hasSelection()) {
          const currentSelection = term.getSelection()
          selectionRef.current = currentSelection
          selectionLabelRef.current = terminalSelectionLabel(term, shellNameRef.current, currentSelection)
        } else {
          selectionRef.current = ''
          selectionLabelRef.current = ''
        }

        setStatus('open')
        cleanup.push(
          terminalApi.onData(session.id, data => term.write(data)),
          terminalApi.onExit(session.id, sessionExit => {
            const { code, signal } = sessionExit
            setStatus('closed')
            term.write(`\r\n[terminal exited${signal ? `: ${signal}` : code !== null ? `: ${code}` : ''}]\r\n`)
          })
        )
        window.requestAnimationFrame(fitAndResize)
      })
      .catch(error => {
        setStatus('closed')
        term.write(`Terminal failed to start: ${error instanceof Error ? error.message : String(error)}\r\n`)
      })

    return () => {
      disposed = true
      cleanup.forEach(run => run())

      const id = sessionIdRef.current
      sessionIdRef.current = null

      if (id) {
        void terminalApi.dispose(id)
      }

      term.dispose()
      termRef.current = null
      shellNameRef.current = 'shell'
      selectionRef.current = ''
      selectionLabelRef.current = ''
    }
  }, [addSelectionToChat, cwd])

  return {
    addSelectionToChat,
    hostRef,
    selection,
    selectionStyle,
    shellName,
    status
  }
}
