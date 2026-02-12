'use client';

import Image from 'next/image';
import type { ChangeEvent } from 'react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

type ConversationMessage = {
  id: string;
  role: string;
  text: string;
  images: string[];
  createdAt: number;
  updatedAt: number;
  local?: boolean;
};

type EventLogEntry = {
  id: string;
  type: string;
  title: string;
  description?: string;
  severity: 'info' | 'warn' | 'error';
  ts: number;
};

type ResponseMode = 'synced' | 'fast';
type PipelineNodeState = 'idle' | 'active' | 'done' | 'error';
type PipelineState = {
  capture: PipelineNodeState;
  agent: PipelineNodeState;
  lipsync: PipelineNodeState;
  playback: PipelineNodeState;
};

type SessionSummary = {
  userTurns: number;
  assistantTurns: number;
  toolCalls: number;
  lastAssistantReply: string;
  memoryKey: string;
  responseMode: ResponseMode;
  endedAt: number;
};

const DEFAULT_WS_BASE = process.env.NEXT_PUBLIC_REALTIME_WS_URL ?? 'ws://localhost:8000';
const CHUNK_SIZE = 60_000;
const MAX_EVENTS = 150;
const MAX_MESSAGES = 200;
const MAX_IMAGE_UPLOAD_MB = 8;
const MAX_IMAGE_UPLOAD_BYTES = MAX_IMAGE_UPLOAD_MB * 1024 * 1024;
const ACCEPTED_IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
const INITIAL_PIPELINE_STATE: PipelineState = {
  capture: 'idle',
  agent: 'idle',
  lipsync: 'idle',
  playback: 'idle',
};
type PersonaKey = 'joi' | 'officer_k' | 'officer_j';
const PERSONA_DEFAULT_THINKING_VIDEO: Record<PersonaKey, string> = {
  joi: '/joi-thinking.mp4',
  officer_k: '/officer_k-thinking.mp4',
  officer_j: '/officer_j-thinking.mp4',
};

const randomId = (prefix: string) => `${prefix}_${Math.random().toString(36).slice(2, 10)}`;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === 'object';

const formatTimestamp = (ts: number) => new Date(ts).toLocaleTimeString();

async function fileToDataUrl(file: File): Promise<string> {
  return await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === 'string') {
        resolve(reader.result);
      } else {
        reject(new Error('Unsupported file reader result.'));
      }
    };
    reader.onerror = () => reject(reader.error ?? new Error('Failed to read file.'));
    reader.readAsDataURL(file);
  });
}

async function prepareImageDataUrl(file: File): Promise<string> {
  const original = await fileToDataUrl(file);
  if (typeof window === 'undefined') {
    return original;
  }
  try {
    return await new Promise<string>((resolve) => {
      const image = new window.Image();
      image.decoding = 'async';
      image.onload = () => {
        try {
          const maxDim = 1024;
          const maxSide = Math.max(image.width, image.height);
          const scale = maxSide > maxDim ? maxDim / maxSide : 1;
          const width = Math.max(1, Math.round(image.width * scale));
          const height = Math.max(1, Math.round(image.height * scale));
          const canvas = document.createElement('canvas');
          canvas.width = width;
          canvas.height = height;
          const ctx = canvas.getContext('2d');
          if (!ctx) {
            resolve(original);
            return;
          }
          ctx.drawImage(image, 0, 0, width, height);
          resolve(canvas.toDataURL('image/jpeg', 0.85));
        } catch (error) {
          console.warn('Image resize failed; sending original.', error);
          resolve(original);
        }
      };
      image.onerror = () => resolve(original);
      image.src = original;
    });
  } catch (error) {
    console.warn('Image processing failed; sending original.', error);
    return original;
  }
}

function parseHistoryMessageItem(item: unknown): ConversationMessage | null {
  if (!isRecord(item)) {
    return null;
  }
  const type = typeof item.type === 'string' ? item.type : null;
  if (type !== 'message') {
    return null;
  }
  const id = typeof item.item_id === 'string' && item.item_id.length > 0 ? item.item_id : randomId('msg');
  const role = typeof item.role === 'string' && item.role.length > 0 ? item.role : 'assistant';
  const textParts: string[] = [];
  const images: string[] = [];
  if (Array.isArray(item.content)) {
    for (const part of item.content) {
      if (!isRecord(part)) {
        continue;
      }
      const partType = typeof part.type === 'string' ? part.type : null;
      if ((partType === 'text' || partType === 'input_text') && typeof part.text === 'string') {
        textParts.push(part.text);
      } else if ((partType === 'input_audio' || partType === 'audio') && typeof part.transcript === 'string') {
        textParts.push(part.transcript);
      } else if (partType === 'input_image') {
        const url = typeof part.image_url === 'string' ? part.image_url : typeof part.url === 'string' ? part.url : null;
        if (url) {
          images.push(url);
        }
      }
    }
  }
  const createdAt = typeof item.created_at === 'string' ? Date.parse(item.created_at) : Date.now();
  const text = textParts.join('').trim();
  return {
    id,
    role,
    text,
    images,
    createdAt: Number.isFinite(createdAt) ? createdAt : Date.now(),
    updatedAt: Date.now(),
  };
}

export default function Home() {
  const [sessionId, setSessionId] = useState<string>('');
  const [memoryKey, setMemoryKey] = useState<string>('');
  const [persona, setPersona] = useState<PersonaKey>('joi');
  const [thinkingVideo, setThinkingVideo] = useState<string>(PERSONA_DEFAULT_THINKING_VIDEO['joi']);
  const [videoUrl, setVideoUrl] = useState<string>('');
  useEffect(() => {
    // Generate a stable client-only session id to avoid SSR/client mismatch
    setSessionId(randomId('session'));
  }, []);
  useEffect(() => {
    // Stable per-browser memory key lets backend persist lightweight personalization.
    if (typeof window === 'undefined') {
      return;
    }
    const storageKey = 'deckard_memory_key';
    const stored = window.localStorage.getItem(storageKey);
    if (stored && stored.trim()) {
      setMemoryKey(stored.trim());
      return;
    }
    const generated = randomId('memory');
    window.localStorage.setItem(storageKey, generated);
    setMemoryKey(generated);
  }, []);
  const wsBase = useMemo(() => {
    const value = DEFAULT_WS_BASE.trim();
    return value.endsWith('/') ? value.slice(0, -1) : value;
  }, []);

  const buildWsUrl = useCallback((base: string, id: string, key?: string) => {
    let b = base.trim();
    if (b.endsWith('/')) b = b.slice(0, -1);
    const core = b.endsWith('/ws') ? `${b}/${id}` : `${b}/ws/${id}`;
    if (!key || !key.trim()) {
      return core;
    }
    return `${core}?memory_key=${encodeURIComponent(key.trim())}`;
  }, []);

  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const isCapturingRef = useRef(false);
  const isMutedRef = useRef(false);
  const playbackContextRef = useRef<AudioContext | null>(null);
  const playbackCursorRef = useRef(0);
  const playbackSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const coordinatedAudioBufferRef = useRef<string[]>([]);
  const isCoordinatedModeRef = useRef(false);
  const uploadResetTimerRef = useRef<number | null>(null);

  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [isCapturing, setIsCapturing] = useState(false);
  const [statusText, setStatusText] = useState('Disconnected');
  const [lastError, setLastError] = useState<string | null>(null);
  const [isThinking, setIsThinking] = useState(false);
  const [messages, setMessages] = useState<ConversationMessage[]>([]);
  const messagesMapRef = useRef<Record<string, ConversationMessage>>({});
  const [events, setEvents] = useState<EventLogEntry[]>([]);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [promptText, setPromptText] = useState('Please describe this image.');
  const [chatInput, setChatInput] = useState('');
  const [responseMode, setResponseMode] = useState<ResponseMode>('synced');
  const [pipeline, setPipeline] = useState<PipelineState>(INITIAL_PIPELINE_STATE);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadStatus, setUploadStatus] = useState('');
  const [sessionSummary, setSessionSummary] = useState<SessionSummary | null>(null);
  const [liveAnnouncement, setLiveAnnouncement] = useState('');
  const [userInteracted, setUserInteracted] = useState(false);
  const [leftPanelOpen, setLeftPanelOpen] = useState(false);
  const [rightPanelOpen, setRightPanelOpen] = useState(false);

  useEffect(() => {
    isMutedRef.current = isMuted;
  }, [isMuted]);

  useEffect(() => {
    setThinkingVideo((previous) => {
      if (typeof previous === 'string' && previous.startsWith(`/${persona}`)) {
        return previous;
      }
      return PERSONA_DEFAULT_THINKING_VIDEO[persona];
    });
  }, [persona]);

  const updatePipeline = useCallback((patch: Partial<PipelineState>) => {
    setPipeline((previous) => ({ ...previous, ...patch }));
  }, []);

  useEffect(() => {
    const micLive = isCapturing && !isMuted;
    updatePipeline({
      capture: micLive ? 'active' : isConnected ? 'idle' : 'idle',
    });
  }, [isCapturing, isConnected, isMuted, updatePipeline]);

  useEffect(() => {
    const parts: string[] = [];
    if (statusText) parts.push(`Connection: ${statusText}`);
    if (uploadStatus) parts.push(`Upload: ${uploadStatus}`);
    if (lastError) parts.push(`Error: ${lastError}`);
    setLiveAnnouncement(parts.join('. '));
  }, [lastError, statusText, uploadStatus]);

  const logEvent = useCallback(
    (type: string, title: string, description?: string, severity: 'info' | 'warn' | 'error' = 'info') => {
      const entry: EventLogEntry = {
        id: randomId('evt'),
        type,
        title,
        description,
        severity,
        ts: Date.now(),
      };
      setEvents((prev) => {
        const next = [entry, ...prev];
        return next.length > MAX_EVENTS ? next.slice(0, MAX_EVENTS) : next;
      });
      if (severity === 'error') {
        setLastError(description ?? title);
      }
    },
    []
  );

  const upsertMessage = useCallback((incoming: ConversationMessage) => {
    const existing = messagesMapRef.current[incoming.id];
    const createdAt = existing?.createdAt ?? incoming.createdAt;
    const text = incoming.text && incoming.text.trim().length > 0 ? incoming.text : existing?.text ?? '';
    const images = incoming.images.length > 0 ? incoming.images : existing?.images ?? [];
    const role = incoming.role || existing?.role || 'assistant';
    const next: ConversationMessage = {
      id: incoming.id,
      role,
      text,
      images,
      createdAt,
      updatedAt: Date.now(),
      local: incoming.local ?? existing?.local,
    };
    messagesMapRef.current[incoming.id] = next;
    const all = Object.values(messagesMapRef.current).sort((a, b) => a.createdAt - b.createdAt);
    const trimmed = all.length > MAX_MESSAGES ? all.slice(all.length - MAX_MESSAGES) : all;
    if (trimmed.length !== all.length) {
      const map: Record<string, ConversationMessage> = {};
      for (const msg of trimmed) {
        map[msg.id] = msg;
      }
      messagesMapRef.current = map;
    }
    setMessages(trimmed);
  }, []);

  const ingestHistory = useCallback(
    (history: unknown[]) => {
      if (!Array.isArray(history)) {
        return;
      }
      for (const item of history) {
        const parsed = parseHistoryMessageItem(item);
        if (!parsed) {
          continue;
        }
        parsed.local = false;
        upsertMessage(parsed);
      }
    },
    [upsertMessage]
  );

  const ingestItem = useCallback(
    (item: unknown) => {
      const parsed = parseHistoryMessageItem(item);
      if (!parsed) {
        return;
      }
      parsed.local = false;
      upsertMessage(parsed);
    },
    [upsertMessage]
  );

  const appendLocalMessage = useCallback(
    (message: { role: string; text: string; images?: string[] }) => {
      const now = Date.now();
      upsertMessage({
        id: randomId('local'),
        role: message.role,
        text: message.text,
        images: message.images ?? [],
        createdAt: now,
        updatedAt: now,
        local: true,
      });
    },
    [upsertMessage]
  );

  const buildSessionSummary = useCallback((): SessionSummary | null => {
    if (messages.length === 0) {
      return null;
    }
    const userTurns = messages.filter((message) => message.role === 'user').length;
    const assistantTurns = messages.filter((message) => message.role === 'assistant').length;
    const toolCalls = events.filter((event) => event.type === 'tool').length;
    const lastAssistant = [...messages].reverse().find((message) => message.role === 'assistant');
    return {
      userTurns,
      assistantTurns,
      toolCalls,
      lastAssistantReply: lastAssistant?.text?.slice(0, 200) ?? 'No assistant reply captured.',
      memoryKey: memoryKey || 'session-scoped',
      responseMode,
      endedAt: Date.now(),
    };
  }, [events, memoryKey, messages, responseMode]);

  const sendPayload = useCallback((payload: Record<string, unknown>) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      return false;
    }
    ws.send(JSON.stringify(payload));
    return true;
  }, []);

  const stopPlayback = useCallback(() => {
    for (const source of playbackSourcesRef.current) {
      try {
        source.stop();
      } catch {}
    }
    playbackSourcesRef.current.clear();
    playbackCursorRef.current = 0;
    const context = playbackContextRef.current;
    if (context) {
      playbackContextRef.current = null;
      context.close().catch(() => undefined);
    }
  }, []);

  const playPcm16Base64 = useCallback(
    async (base64Audio: string) => {
      if (typeof window === 'undefined' || !base64Audio) {
        return;
      }
      let context = playbackContextRef.current;
      if (!context || context.state === 'closed') {
        context = new AudioContext({ sampleRate: 24_000, latencyHint: 'interactive' });
        playbackContextRef.current = context;
        playbackCursorRef.current = context.currentTime;
      }
      if (context.state === 'suspended') {
        try {
          await context.resume();
        } catch {}
      }

      try {
        const binary = window.atob(base64Audio);
        const byteLength = binary.length;
        if (byteLength < 2 || byteLength % 2 !== 0) {
          return;
        }
        const bytes = new Uint8Array(byteLength);
        for (let i = 0; i < byteLength; i += 1) {
          bytes[i] = binary.charCodeAt(i);
        }
        const view = new DataView(bytes.buffer);
        const sampleCount = byteLength / 2;
        const float32 = new Float32Array(sampleCount);
        for (let i = 0; i < sampleCount; i += 1) {
          const sample = view.getInt16(i * 2, true);
          float32[i] = sample / 32768;
        }

        const buffer = context.createBuffer(1, sampleCount, 24_000);
        buffer.copyToChannel(float32, 0);
        const source = context.createBufferSource();
        source.buffer = buffer;
        source.connect(context.destination);
        const startAt = Math.max(context.currentTime + 0.01, playbackCursorRef.current || context.currentTime);
        source.start(startAt);
        playbackCursorRef.current = startAt + buffer.duration;
        playbackSourcesRef.current.add(source);
        source.onended = () => {
          playbackSourcesRef.current.delete(source);
        };
      } catch (error) {
        logEvent('media', 'Audio playback failed', error instanceof Error ? error.message : 'Invalid audio payload', 'warn');
      }
    },
    [logEvent]
  );




  const startCapture = useCallback(async () => {
    if (typeof window === 'undefined') {
      throw new Error('Window unavailable.');
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('Microphone access requires HTTPS or a supported browser.');
    }
    if (isCapturingRef.current) {
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 24_000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      const audioContext = new AudioContext({ sampleRate: 24_000, latencyHint: 'interactive' });
      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      source.connect(processor);
      processor.connect(audioContext.destination);
      processor.onaudioprocess = (event) => {
        if (isMutedRef.current) {
          return;
        }
        const ws = wsRef.current;
        if (!ws || ws.readyState !== WebSocket.OPEN) {
          return;
        }
        const inputBuffer = event.inputBuffer.getChannelData(0);
        const int16Buffer = new Int16Array(inputBuffer.length);
        for (let i = 0; i < inputBuffer.length; i += 1) {
          int16Buffer[i] = Math.max(-32768, Math.min(32767, inputBuffer[i] * 32768));
        }
        ws.send(
          JSON.stringify({
            type: 'audio',
            data: Array.from(int16Buffer),
          })
        );
      };
      audioContextRef.current = audioContext;
      processorRef.current = processor;
      streamRef.current = stream;
      isCapturingRef.current = true;
      setIsCapturing(true);
      logEvent('media', 'Microphone streaming', 'Audio capture is live.');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown microphone error';
      logEvent('media', 'Unable to start capture', message, 'error');
      throw error;
    }
  }, [logEvent]);

  const stopCapture = useCallback(() => {
    isCapturingRef.current = false;
    setIsCapturing(false);
    const processor = processorRef.current;
    if (processor) {
      try {
        processor.disconnect();
      } catch (error) {
        console.warn('Failed to disconnect processor.', error);
      }
      processor.onaudioprocess = null;
      processorRef.current = null;
    }
    const audioContext = audioContextRef.current;
    if (audioContext) {
      audioContextRef.current = null;
      audioContext.close().catch(() => undefined);
    }
    const stream = streamRef.current;
    if (stream) {
      for (const track of stream.getTracks()) {
        track.stop();
      }
      streamRef.current = null;
    }
  }, []);

  const handleRealtimeEvent = useCallback(
    (event: unknown) => {
      if (!isRecord(event)) {
        return;
      }
      const type = typeof event.type === 'string' ? event.type : null;
      switch (type) {
        case 'talk_video': {
          const url = typeof event.url === 'string' ? event.url : '';
          const coordinated = typeof event.coordinated === 'boolean' ? event.coordinated : false;

          if (url) {
            setVideoUrl(url);
            setIsThinking(false);
            updatePipeline({ lipsync: 'done', playback: 'active', agent: 'done' });
            if (coordinated) {
              // Play buffered audio in sync with video (chunks already queued in order via playbackCursorRef).
              const chunks = coordinatedAudioBufferRef.current;
              coordinatedAudioBufferRef.current = [];
              isCoordinatedModeRef.current = false;
              for (const chunk of chunks) {
                void playPcm16Base64(chunk);
              }
              logEvent('video', 'Coordinated video ready', `Video synchronized: ${url}`);
            } else {
              logEvent('video', 'Lip-sync video ready', url);
            }
          } else {
            logEvent('video', 'Lip-sync status', String(event.status ?? 'unknown'));
          }
          break;
        }
        case 'talk_error': {
          const error = typeof event.error === 'string' ? event.error : 'Unknown lip-sync error';
          setIsThinking(false);
          updatePipeline({ lipsync: 'error', playback: 'active' });
          logEvent('error', 'Lip-sync generation failed', error, 'error');
          break;
        }
        case 'audio': {
          const payload = typeof event.audio === 'string' ? event.audio : null;
          if (payload) {
            updatePipeline({ playback: 'active' });
            if (isCoordinatedModeRef.current) {
              coordinatedAudioBufferRef.current.push(payload);
            } else {
              void playPcm16Base64(payload);
            }
          }
          break;
        }
        case 'audio_end': {
          // Keep cursor near current clock to avoid unbounded drift over long sessions.
          const context = playbackContextRef.current;
          if (context) {
            playbackCursorRef.current = Math.max(playbackCursorRef.current, context.currentTime);
          }
          updatePipeline({ playback: 'done', agent: 'done' });
          break;
        }
        case 'history_updated': {
          if (Array.isArray(event.history)) {
            ingestHistory(event.history);
          }
          break;
        }
        case 'history_added': {
          ingestItem(event.item);
          break;
        }
        case 'tool_start': {
          const tool = typeof event.tool === 'string' ? event.tool : 'tool';
          updatePipeline({ agent: 'active' });
          logEvent('tool', `Tool running`, `Started ${tool}`);
          break;
        }
        case 'tool_end': {
          const tool = typeof event.tool === 'string' ? event.tool : 'tool';
          const output = typeof event.output === 'string' ? event.output : 'no output';
          updatePipeline({ agent: 'done' });
          logEvent('tool', `Tool completed`, `${tool}: ${output}`);
          break;
        }
        case 'handoff': {
          const fromAgent = isRecord(event.from_agent) && typeof event.from_agent.name === 'string' ? event.from_agent.name : null;
          const toAgent = isRecord(event.to_agent) && typeof event.to_agent.name === 'string' ? event.to_agent.name : null;
          const from = typeof event.from === 'string' ? event.from : fromAgent ?? 'agent';
          const to = typeof event.to === 'string' ? event.to : toAgent ?? 'agent';
          logEvent('handoff', 'Agent handoff', `${from} → ${to}`);
          break;
        }
        case 'client_info': {
          const info = typeof event.info === 'string' ? event.info : 'client info';

          // Handle special response processing notifications
          if (info === 'response_processing') {
            const message = typeof event.message === 'string' ? event.message : 'Generating response...';
            const video = typeof event.video === 'string' ? event.video : null;
            if (video) {
              setThinkingVideo(video);
            }
            setIsThinking(true);
            updatePipeline({ agent: 'active', lipsync: 'active' });
            logEvent('response', 'Processing Response', message);
          } else if (info === 'persona_mood_update') {
            const personaRaw = typeof event.persona === 'string' ? event.persona : null;
            const video = typeof event.video === 'string' ? event.video : null;
            const sentiment = typeof event.sentiment === 'string' ? event.sentiment : undefined;

            if (personaRaw === 'joi' || personaRaw === 'officer_k' || personaRaw === 'officer_j') {
              const personaFromEvent: PersonaKey = personaRaw;
              setPersona(personaFromEvent);
              setThinkingVideo(video ?? PERSONA_DEFAULT_THINKING_VIDEO[personaFromEvent]);
              logEvent('persona', 'Persona mood updated', `${personaFromEvent} · ${sentiment ?? 'unknown'}`);
            } else if (video) {
              setThinkingVideo(video);
            }
          } else if (info === 'persona_set') {
            const personaRaw = typeof event.persona === 'string' ? event.persona : null;
            if (personaRaw === 'joi' || personaRaw === 'officer_k' || personaRaw === 'officer_j') {
              const personaFromEvent: PersonaKey = personaRaw;
              setPersona(personaFromEvent);
              setThinkingVideo(PERSONA_DEFAULT_THINKING_VIDEO[personaFromEvent]);
            }
          } else if (info === 'did_talk_start') {
            setIsThinking(true);
            updatePipeline({ lipsync: 'active' });
            logEvent('video', 'Video generation started');
          } else if (info === 'coordinated_audio_start') {
            isCoordinatedModeRef.current = true;
            coordinatedAudioBufferRef.current = [];
            setIsThinking(true);
            updatePipeline({ playback: 'idle', lipsync: 'active', agent: 'active' });
            logEvent('video', 'Coordinated playback', 'Buffering audio for sync');
          } else if (info === 'response_mode_set') {
            const mode = typeof event.mode === 'string' ? event.mode : '';
            if (mode === 'synced' || mode === 'fast') {
              setResponseMode(mode);
              logEvent('session', 'Response mode set', mode === 'synced' ? 'Synced avatar mode' : 'Fast audio mode');
            }
          } else if (info === 'text_enqueued') {
            const chars = typeof event.chars === 'number' ? event.chars : 0;
            logEvent('session', 'Text message enqueued', `${chars} characters sent`);
          } else {
            logEvent('client', `Client info`, info);
          }
          break;
        }
        case 'guardrail_tripped': {
          const names = Array.isArray(event.guardrail_results)
            ? event.guardrail_results
                .map((result) => (isRecord(result) && typeof result.name === 'string' ? result.name : null))
                .filter((value): value is string => Boolean(value))
                .join(', ')
            : null;
          logEvent('guardrail', 'Guardrail triggered', names || 'Guardrail threshold met', 'warn');
          break;
        }
        case 'input_audio_timeout_triggered': {
          if (sendPayload({ type: 'commit_audio' })) {
            updatePipeline({ agent: 'active' });
            logEvent('session', 'Committed audio buffer');
          }
          break;
        }
        case 'raw_model_event': {
          const rawType = isRecord(event.raw_model_event) && typeof event.raw_model_event.type === 'string' ? event.raw_model_event.type : 'raw';
          logEvent('model', `Model event`, rawType);
          break;
        }
        case 'error': {
          const message = event.error ? String(event.error) : 'Unknown realtime error';
          updatePipeline({ agent: 'error' });
          logEvent('error', 'Realtime error', message, 'error');
          break;
        }
        default: {
          const label = type ?? 'event';
          logEvent('event', `Event: ${label}`);
        }
      }
    },
    [ingestHistory, ingestItem, logEvent, playPcm16Base64, sendPayload, updatePipeline]
  );

  const openConnection = useCallback(() => {
    if (isConnecting || isConnected) {
      return;
    }
    setStatusText('Connecting...');
    setIsConnecting(true);
    setLastError(null);
    const effectiveMemoryKey = memoryKey || randomId('memory');
    if (!memoryKey) {
      setMemoryKey(effectiveMemoryKey);
      if (typeof window !== 'undefined') {
        window.localStorage.setItem('deckard_memory_key', effectiveMemoryKey);
      }
    }
    const effectiveId = sessionId || randomId('session');
    if (!sessionId) setSessionId(effectiveId);
    const url = buildWsUrl(wsBase, effectiveId, effectiveMemoryKey);
    const socket = new WebSocket(url);
    wsRef.current = socket;
    setSessionSummary(null);
    setPipeline(INITIAL_PIPELINE_STATE);
    logEvent('session', 'Connecting', `Dialing ${url}`);

    socket.onopen = async () => {
      setIsConnecting(false);
      setIsConnected(true);
      setStatusText('Connected');
      logEvent('session', 'Connected', `Session ${sessionId}`);
      try {
        await startCapture();
      } catch (error) {
        console.warn('Microphone capture failed after connection.', error);
      }
      // Send initial persona selection to backend
      try {
        sendPayload({ type: 'set_persona', persona });
        sendPayload({ type: 'set_response_mode', mode: responseMode });
      } catch {}
    };

    socket.onmessage = (event) => {
      try {
        const payload: unknown = JSON.parse(event.data);
        handleRealtimeEvent(payload);
      } catch (error) {
        logEvent('error', 'Malformed realtime payload', error instanceof Error ? error.message : 'Unknown parse error', 'error');
      }
    };

    socket.onerror = (event) => {
      console.error('WebSocket error', event);
      logEvent('error', 'WebSocket error', 'Check backend logs for details.', 'error');
    };

    socket.onclose = (event) => {
      const reason = event.reason || `Socket closed (${event.code})`;
      logEvent('session', 'Disconnected', reason, event.wasClean ? 'info' : 'warn');
      setIsConnected(false);
      setIsConnecting(false);
      setStatusText('Disconnected');
      stopCapture();
      stopPlayback();
      setPipeline((previous) => ({ ...previous, capture: 'idle' }));
      setSessionSummary(buildSessionSummary());
      wsRef.current = null;
    };
  }, [buildSessionSummary, buildWsUrl, handleRealtimeEvent, isConnected, isConnecting, logEvent, memoryKey, persona, responseMode, sendPayload, sessionId, startCapture, stopCapture, stopPlayback, wsBase]);

  const closeConnection = useCallback(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.close();
    } else {
      wsRef.current = null;
      setIsConnected(false);
      setIsConnecting(false);
      setStatusText('Disconnected');
      stopCapture();
      stopPlayback();
      setPipeline((previous) => ({ ...previous, capture: 'idle' }));
      setSessionSummary(buildSessionSummary());
    }
  }, [buildSessionSummary, stopCapture, stopPlayback]);

  const toggleMute = useCallback(() => {
    setIsMuted((prev) => {
      const next = !prev;
      logEvent('media', next ? 'Microphone muted' : 'Microphone live');
      return next;
    });
  }, [logEvent]);

  const interrupt = useCallback(() => {
    if (sendPayload({ type: 'interrupt' })) {
      stopPlayback();
      setIsThinking(false);
      logEvent('session', 'Interrupt sent', 'Requested model to stop playback');
    }
  }, [logEvent, sendPayload, stopPlayback]);

  const sendTextMessage = useCallback(() => {
    const text = chatInput.trim();
    if (!text) {
      return;
    }
    if (!isConnected) {
      logEvent('session', 'Text not sent', 'Connect before sending a text message.', 'warn');
      return;
    }
    appendLocalMessage({ role: 'user', text });
    const sent = sendPayload({ type: 'text', text });
    if (!sent) {
      logEvent('session', 'Text not sent', 'Realtime socket is not open.', 'warn');
      return;
    }
    updatePipeline({ agent: 'active' });
    setIsThinking(true);
    setChatInput('');
    logEvent('session', 'Text message sent');
  }, [appendLocalMessage, chatInput, isConnected, logEvent, sendPayload, updatePipeline]);

  const applyResponseMode = useCallback(
    (mode: ResponseMode) => {
      setResponseMode(mode);
      if (isConnected) {
        sendPayload({ type: 'set_response_mode', mode });
      }
      logEvent(
        'session',
        'Response mode selected',
        mode === 'synced'
          ? 'Synced avatar mode prioritizes realism.'
          : 'Fast mode prioritizes lower perceived latency.'
      );
    },
    [isConnected, logEvent, sendPayload]
  );

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const isTyping =
        !!target &&
        (target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.isContentEditable);
      if (isTyping) {
        return;
      }
      if (event.key.toLowerCase() === 'm' && isConnected) {
        event.preventDefault();
        toggleMute();
      } else if (event.key.toLowerCase() === 'i' && isConnected) {
        event.preventDefault();
        fileInputRef.current?.click();
      } else if (event.key === 'Escape' && isConnected) {
        event.preventDefault();
        interrupt();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [interrupt, isConnected, toggleMute]);

  const handleFileSelected = useCallback(
    async (file: File | null) => {
      if (!file) {
        return;
      }
      if (!ACCEPTED_IMAGE_TYPES.includes(file.type)) {
        setUploadProgress(0);
        setUploadStatus('Unsupported image type. Use JPG, PNG, or WebP.');
        logEvent('image', 'Image rejected', 'Unsupported file format. Use JPG, PNG, or WebP.', 'warn');
        return;
      }
      if (file.size > MAX_IMAGE_UPLOAD_BYTES) {
        setUploadProgress(0);
        setUploadStatus(`Image too large. Limit is ${MAX_IMAGE_UPLOAD_MB} MB.`);
        logEvent(
          'image',
          'Image rejected',
          `Selected file is ${(file.size / 1_048_576).toFixed(2)} MB. Limit is ${MAX_IMAGE_UPLOAD_MB} MB.`,
          'warn'
        );
        return;
      }
      try {
        setUploadStatus('Preparing image...');
        setUploadProgress(10);
        const dataUrl = await prepareImageDataUrl(file);
        const prompt = promptText.trim();
        appendLocalMessage({
          role: 'user',
          text: prompt || 'Please describe this image.',
          images: [dataUrl],
        });
        const ws = wsRef.current;
        if (!ws || ws.readyState !== WebSocket.OPEN) {
          logEvent('image', 'Image not sent', 'Connect before uploading an image.', 'warn');
          setUploadProgress(0);
          setUploadStatus('Connect first to send an image.');
          return;
        }
        setUploadStatus('Uploading image...');
        sendPayload({ type: 'interrupt' });
        const imageId = randomId('img');
        const promptPayload = prompt || 'Please describe this image.';
        sendPayload({ type: 'image_start', id: imageId, text: promptPayload });
        const totalChunks = Math.max(1, Math.ceil(dataUrl.length / CHUNK_SIZE));
        for (let index = 0; index < dataUrl.length; index += CHUNK_SIZE) {
          sendPayload({ type: 'image_chunk', id: imageId, chunk: dataUrl.slice(index, index + CHUNK_SIZE) });
          const chunkNumber = Math.floor(index / CHUNK_SIZE) + 1;
          const progress = 10 + Math.round((chunkNumber / totalChunks) * 85);
          setUploadProgress(Math.min(95, progress));
        }
        sendPayload({ type: 'image_end', id: imageId });
        setUploadProgress(100);
        setUploadStatus('Image sent to AI.');
        logEvent('image', 'Image enqueued', `Sent ${(dataUrl.length / 1_024).toFixed(1)} KB payload`);
        if (uploadResetTimerRef.current) {
          window.clearTimeout(uploadResetTimerRef.current);
        }
        uploadResetTimerRef.current = window.setTimeout(() => {
          setUploadProgress(0);
          setUploadStatus('');
        }, 3500);
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Unknown image processing error';
        setUploadProgress(0);
        setUploadStatus('Image processing failed.');
        logEvent('error', 'Image processing failed', message, 'error');
      } finally {
        if (fileInputRef.current) {
          fileInputRef.current.value = '';
        }
      }
    },
    [appendLocalMessage, logEvent, promptText, sendPayload]
  );

  const handleFileInputChange = useCallback(
    async (event: ChangeEvent<HTMLInputElement>) => {
      const files = event.target.files;
      await handleFileSelected(files && files.length > 0 ? files[0] : null);
    },
    [handleFileSelected]
  );

  const personaImage = useMemo(() => {
    switch (persona) {
      case 'officer_k':
        return '/officer_k.png';
      case 'officer_j':
        return '/officer_j.png';
      case 'joi':
      default:
        return '/joi.png';
    }
  }, [persona]);

  const personaThinkingVideo = useMemo(
    () => thinkingVideo || PERSONA_DEFAULT_THINKING_VIDEO[persona],
    [persona, thinkingVideo]
  );

  useEffect(() => {
    return () => {
      if (uploadResetTimerRef.current) {
        window.clearTimeout(uploadResetTimerRef.current);
      }
      const ws = wsRef.current;
      if (ws) {
        ws.close();
      }
      stopCapture();
      stopPlayback();
    };
  }, [stopCapture, stopPlayback]);

  const isMicLive = isCapturing && !isMuted;

  return (
    <div className="relative min-h-screen overflow-hidden text-stone-100">
      <div className="sr-only" aria-live="polite" aria-atomic="true">
        {liveAnnouncement}
      </div>
      <div className="pointer-events-none absolute inset-0 -z-20">
        <Image
          src="/website background.png"
          alt="Deckard ambient backdrop"
          fill
          priority
          className="object-cover"
        />
      </div>
      <div className="pointer-events-none absolute inset-0 -z-10">
        <div className="absolute inset-0 bg-gradient-to-br from-stone-950/92 via-stone-900/78 to-stone-950/94" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,_rgba(214,211,209,0.24),_transparent_58%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_bottom,_rgba(120,113,108,0.2),_transparent_60%)]" />
        <div className="absolute inset-0 bg-[linear-gradient(120deg,_rgba(250,250,249,0.08)_0%,_rgba(41,37,36,0.65)_55%,_rgba(12,10,9,0.78)_100%)]" />
      </div>

      {/* Left Panel - Conversation */}
      <div
        className={`fixed left-0 top-0 z-20 h-full w-80 transform bg-stone-950/85 backdrop-blur-2xl transition-transform duration-300 ease-in-out ${
        leftPanelOpen ? 'translate-x-0' : '-translate-x-full'
      }`}
      >
        <div className="h-full border-r border-stone-500/30 p-6">
          <div className="flex items-center justify-between text-[0.65rem] font-semibold uppercase tracking-[0.35em] text-stone-400">
            <div className="flex items-center gap-2">
              <span>Conversation</span>
              <span>·</span>
              <span>{messages.length} messages</span>
            </div>
            <button
              onClick={() => setLeftPanelOpen(false)}
              className="group flex h-8 w-8 items-center justify-center rounded-full border border-stone-500/30 transition-all hover:border-stone-400/50 hover:bg-stone-900/40"
            >
              <svg className="h-4 w-4 text-stone-400 transition-colors group-hover:text-stone-100" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          <div className="mt-4 flex max-h-[calc(100vh-8rem)] flex-col gap-4 overflow-y-auto pr-2 text-sm [scrollbar-color:rgba(168,162,158,0.35)_transparent]">
            {messages.length === 0 ? (
              <p className="rounded-2xl border border-dashed border-stone-500/30 bg-stone-950/40 px-4 py-6 text-center text-stone-500">
                Initiate a connection to populate the conversational thread.
              </p>
            ) : (
              messages.map((message) => (
                <div key={message.id} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div
                    className={`max-w-[85%] rounded-3xl border px-4 py-3 text-sm shadow-[0_25px_80px_rgba(15,23,42,0.45)] ${
                      message.role === 'user'
                        ? 'border-emerald-400/50 bg-emerald-400/10 text-emerald-100'
                        : 'border-stone-500/30 bg-stone-900/40 text-stone-100'
                    }`}
                  >
                    <div className="flex flex-col gap-2">
                      {message.images.length > 0 ? (
                        <div className="grid gap-2">
                          {message.images.map((image, index) => (
                            <Image
                              key={index}
                              src={image}
                              alt={`Uploaded ${index + 1}`}
                              width={200}
                              height={200}
                              className="h-auto w-full rounded-2xl border border-stone-400/30 object-cover"
                            />
                          ))}
                        </div>
                      ) : null}
                      {message.text ? <p className="leading-relaxed text-stone-100/90">{message.text}</p> : null}
                      <span className="text-[0.5rem] uppercase tracking-[0.35em] text-stone-400">{message.role}</span>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
          <div className="mt-4 rounded-2xl border border-stone-500/30 bg-stone-950/45 p-3">
            <label htmlFor="chat-composer" className="text-[0.55rem] font-semibold uppercase tracking-[0.35em] text-stone-500">
              Send Text
            </label>
            <textarea
              id="chat-composer"
              value={chatInput}
              onChange={(event) => setChatInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault();
                  sendTextMessage();
                }
              }}
              rows={3}
              disabled={!isConnected}
              placeholder={isConnected ? 'Type a message and press Enter' : 'Connect to send text'}
              className="mt-2 w-full resize-none rounded-xl border border-stone-500/30 bg-stone-900/50 px-3 py-2 text-sm text-stone-100 placeholder:text-stone-500 focus:border-emerald-300/60 focus:outline-none disabled:cursor-not-allowed disabled:opacity-60"
            />
            <div className="mt-2 flex items-center justify-between gap-2">
              <span className="text-[0.65rem] text-stone-500">Enter to send, Shift+Enter for newline</span>
              <button
                type="button"
                onClick={sendTextMessage}
                disabled={!isConnected || chatInput.trim().length === 0}
                className="rounded-full border border-stone-500/30 px-3 py-1.5 text-[0.65rem] font-semibold uppercase tracking-[0.28em] text-stone-200 transition hover:border-emerald-300/50 hover:text-stone-100 disabled:cursor-not-allowed disabled:opacity-50"
              >
                Send
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Right Panel - Realtime Feed */}
      <div
        className={`fixed right-0 top-0 z-20 h-full w-80 transform bg-stone-950/85 backdrop-blur-2xl transition-transform duration-300 ease-in-out ${
          rightPanelOpen ? 'translate-x-0' : 'translate-x-full'
        }`}
      >
        <div className="h-full border-l border-stone-500/30 p-6">
          <div className="flex items-center justify-between text-[0.65rem] font-semibold uppercase tracking-[0.35em] text-stone-400">
            <div className="flex items-center gap-2">
              <span>Realtime Feed</span>
              <button
                className="rounded-full border border-stone-500/30 px-2 py-1 text-[0.5rem] uppercase tracking-[0.35em] text-stone-400 transition hover:border-stone-400/60 hover:text-stone-100"
                onClick={() => setEvents([])}
              >
                Clear
              </button>
            </div>
            <button
              onClick={() => setRightPanelOpen(false)}
              className="group flex h-8 w-8 items-center justify-center rounded-full border border-stone-500/30 transition-all hover:border-stone-400/50 hover:bg-stone-900/40"
            >
              <svg className="h-4 w-4 text-stone-400 transition-colors group-hover:text-stone-100" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          <div className="mt-4 flex max-h-[calc(100vh-8rem)] flex-col gap-3 overflow-y-auto pr-2 text-sm [scrollbar-color:rgba(168,162,158,0.35)_transparent]">
            {events.length === 0 ? (
              <p className="rounded-2xl border border-dashed border-stone-500/30 bg-stone-950/40 px-4 py-6 text-center text-stone-500">
                Streamed tool events and guardrail updates will appear here.
              </p>
            ) : (
              events.map((event) => (
                <div
                  key={event.id}
                  className={`rounded-2xl border px-4 py-3 text-sm shadow-[0_20px_70px_rgba(15,23,42,0.45)] ${
                    event.severity === 'error'
                      ? 'border-rose-500/60 bg-rose-500/10 text-rose-100'
                      : event.severity === 'warn'
                      ? 'border-amber-400/60 bg-amber-400/10 text-amber-100'
                      : 'border-stone-500/30 bg-stone-900/40 text-stone-100'
                  }`}
                >
                  <div className="flex items-center justify-between text-[0.5rem] uppercase tracking-[0.35em] text-stone-400">
                    <span>{event.type}</span>
                    <span>{formatTimestamp(event.ts)}</span>
                  </div>
                  <div className="mt-1 font-semibold text-stone-100">{event.title}</div>
                  {event.description ? <div className="mt-1 text-xs text-stone-200/80">{event.description}</div> : null}
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      <main className="relative z-10 mx-auto flex min-h-screen w-full flex-col items-center px-6 py-8">
        {/* AI DECKARD Title */}
        <div className="mb-12 text-center">
          <h1 className="mt-4 flex items-center justify-center gap-4 text-6xl font-bold tracking-tight text-white sm:text-7xl">
            <Image
              src="/c281291e-d2ca-4240-97bb-93a2526aa38d.png"
              alt="Deckard company logo"
              width={160}
              height={160}
              priority
              className="h-12 w-auto sm:h-16"
            />
            <span className="bg-gradient-to-r from-stone-200 via-stone-300 to-stone-400 bg-clip-text text-transparent drop-shadow-[0_10px_36px_rgba(120,113,108,0.45)]">
              Deckard
            </span>
          </h1>
          <p className="mt-4 mx-auto max-w-2xl text-base text-stone-300 sm:text-lg">
            Create and interact with personalized AI clones
          </p>
          <div className="mt-5 mx-auto max-w-3xl rounded-2xl border border-stone-500/30 bg-stone-900/45 px-5 py-4 text-left text-sm leading-relaxed text-stone-300 shadow-[0_18px_60px_rgba(15,23,42,0.45)]">
            <span className="block text-[0.6rem] font-semibold uppercase tracking-[0.35em] text-stone-400">How It Works</span>
            <p className="mt-2">
              Your voice is streamed to a realtime AI agent, which can use tools and optional uploaded image context to build a response.
              The generated assistant audio is then synchronized with the selected avatar for a conversational lip-synced reply.
            </p>
          </div>
        </div>

        {/* Main Content Area with Avatar */}
        <section className="relative w-full max-w-5xl">
          <div className="flex items-start justify-center gap-8 sm:gap-12">
            {/* Left Toggle Button - Conversation */}
            <button
              onClick={() => setLeftPanelOpen(!leftPanelOpen)}
              className="group flex h-20 w-24 flex-col items-center justify-center self-start rounded-2xl border border-stone-500/30 bg-stone-900/40 backdrop-blur-2xl transition-all hover:border-stone-400/60 hover:bg-stone-900/55 sm:-translate-y-2"
            >
              <div className="mb-2 flex flex-col items-center gap-1">
                <svg className="h-4 w-4 text-stone-300 transition-colors group-hover:text-stone-100" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
                <svg className={`h-3 w-3 text-stone-400 transition-all group-hover:text-stone-100 ${leftPanelOpen ? 'rotate-90' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </div>
              <span className="text-[0.6rem] font-semibold uppercase tracking-[0.3em] text-stone-300 transition-colors group-hover:text-stone-100">
                Chat
              </span>
            </button>

            {/* Central Avatar Area */}
            <div className="relative overflow-hidden rounded-[32px] border border-stone-500/35 bg-stone-900/45 p-8 shadow-[0_35px_140px_rgba(2,6,23,0.65)] backdrop-blur-2xl">
              <div className="flex flex-col gap-4">
                <div className="flex flex-wrap items-center justify-between gap-4 text-[0.65rem] uppercase tracking-[0.35em] text-stone-400">
                  <div className="flex items-center gap-3">
                    <span
                      className={`inline-flex h-3 w-3 rounded-full shadow-[0_0_24px_rgba(34,197,94,0.65)] ${
                        isConnected ? 'bg-emerald-400' : isConnecting ? 'bg-amber-400' : 'bg-stone-600'
                      }`}
                    />
                    <span className="font-semibold text-stone-200">{statusText}</span>
                  </div>
                  <span className={`font-semibold ${isMicLive ? 'text-emerald-200' : isMuted ? 'text-stone-500' : 'text-stone-300'}`}>
                    {isConnected ? (isMicLive ? 'Microphone live' : isMuted ? 'Microphone muted' : 'Microphone idle') : 'Awaiting connection'}
                  </span>
                </div>
                {lastError ? (
                  <div role="alert" aria-live="assertive" className="rounded-2xl border border-rose-500/50 bg-rose-500/10 px-4 py-3 text-xs text-rose-200">
                    {lastError}
                  </div>
                ) : null}
                <div className="rounded-2xl border border-stone-500/30 bg-stone-950/45 px-4 py-3">
                  <div className="text-[0.55rem] font-semibold uppercase tracking-[0.35em] text-stone-500">Pipeline</div>
                  <div className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-4">
                    {([
                      ['Capture', pipeline.capture],
                      ['Agent', pipeline.agent],
                      ['Lip-sync', pipeline.lipsync],
                      ['Playback', pipeline.playback],
                    ] as const).map(([label, state]) => (
                      <div key={label} className="rounded-xl border border-stone-500/25 bg-stone-900/35 px-3 py-2">
                        <div className="text-[0.55rem] uppercase tracking-[0.28em] text-stone-400">{label}</div>
                        <div
                          className={`mt-1 text-xs font-semibold ${
                            state === 'active'
                              ? 'text-emerald-200'
                              : state === 'done'
                              ? 'text-sky-200'
                              : state === 'error'
                              ? 'text-rose-200'
                              : 'text-stone-400'
                          }`}
                        >
                          {state}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="relative mx-auto mt-8 w-full max-w-sm overflow-hidden rounded-[28px] border border-stone-500/35 bg-gradient-to-b from-stone-950/85 via-stone-900/35 to-stone-950/95" data-testid="talking-video-box" style={{ aspectRatio: '9 / 16' }}>
                {videoUrl ? (
                  <video
                    src={videoUrl}
                    autoPlay
                    muted={!userInteracted}
                    playsInline
                    className="h-full w-full object-cover"
                    poster={personaImage}
                  />
                ) : (
                  <Image src={personaImage} alt="Persona" fill className="object-cover" unoptimized />
                )}
                {isThinking && (
                  <div className="absolute inset-0 z-10 overflow-hidden">
                    <video
                      key={personaThinkingVideo}
                      src={personaThinkingVideo}
                      autoPlay
                      loop
                      muted
                      playsInline
                      className="h-full w-full object-cover"
                    />
                  </div>
                )}
              </div>

              <div className="mt-6 flex flex-wrap justify-center gap-2">
                {(['joi', 'officer_k', 'officer_j'] as const).map((key) => (
                  <button
                    key={key}
                    onClick={() => {
                      setPersona(key);
                      sendPayload({ type: 'set_persona', persona: key });
                      logEvent('client', 'Persona selected', key);
                    }}
                    className={`rounded-full border px-4 py-2 text-[0.65rem] font-semibold uppercase tracking-[0.35em] transition ${
                      persona === key
                        ? 'border-emerald-400/70 bg-emerald-400/10 text-emerald-100 shadow-[0_0_24px_rgba(16,185,129,0.35)]'
                        : 'border-stone-500/30 bg-stone-900/40 text-stone-300 hover:text-stone-100'
                    }`}
                  >
                    {key.replace('_', ' ').toUpperCase()}
                  </button>
                ))}
              </div>

              <div className="mt-5 rounded-2xl border border-stone-500/30 bg-stone-950/45 px-4 py-3">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <p className="text-[0.55rem] font-semibold uppercase tracking-[0.35em] text-stone-500">Response Mode</p>
                    <p className="mt-1 text-xs text-stone-400">
                      Choose realism or speed for assistant playback.
                    </p>
                  </div>
                  <div className="inline-flex rounded-full border border-stone-500/30 bg-stone-900/40 p-1">
                    <button
                      type="button"
                      onClick={() => applyResponseMode('synced')}
                      className={`rounded-full px-4 py-2 text-[0.65rem] font-semibold uppercase tracking-[0.28em] transition ${
                        responseMode === 'synced'
                          ? 'bg-emerald-400/20 text-emerald-100'
                          : 'text-stone-300 hover:text-stone-100'
                      }`}
                      aria-pressed={responseMode === 'synced'}
                    >
                      Synced
                    </button>
                    <button
                      type="button"
                      onClick={() => applyResponseMode('fast')}
                      className={`rounded-full px-4 py-2 text-[0.65rem] font-semibold uppercase tracking-[0.28em] transition ${
                        responseMode === 'fast'
                          ? 'bg-emerald-400/20 text-emerald-100'
                          : 'text-stone-300 hover:text-stone-100'
                      }`}
                      aria-pressed={responseMode === 'fast'}
                    >
                      Fast
                    </button>
                  </div>
                </div>
              </div>

              <div className="mt-8 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                <button
                  className={`rounded-full px-5 py-3 text-sm font-semibold uppercase tracking-[0.35em] transition ${
                    isConnected
                      ? 'bg-rose-500/15 text-rose-100 hover:bg-rose-500/25'
                      : 'bg-emerald-400 text-stone-950 hover:bg-emerald-300'
                  } ${isConnecting ? 'opacity-70' : ''}`}
                  onClick={() => {
                    setUserInteracted(true);
                    if (isConnected) {
                      closeConnection();
                    } else {
                      openConnection();
                    }
                  }}
                  disabled={isConnecting}
                >
                  {isConnected ? 'Disconnect' : isConnecting ? 'Connecting...' : 'Connect'}
                </button>
                <button
                  className={`rounded-full border border-stone-500/30 px-5 py-3 text-sm font-semibold uppercase tracking-[0.35em] transition ${
                    isMicLive ? 'bg-emerald-500/10 text-emerald-200' : isMuted ? 'bg-stone-950 text-stone-500' : 'bg-stone-950 text-stone-200'
                  } ${!isConnected ? 'opacity-50' : ''}`}
                  onClick={toggleMute}
                  disabled={!isConnected}
                >
                  {isMicLive ? 'Mic Live' : isMuted ? 'Mic Muted' : 'Enable Mic'}
                </button>
                <button
                  className="rounded-full border border-stone-500/30 px-5 py-3 text-sm font-semibold uppercase tracking-[0.35em] text-stone-200 transition hover:border-emerald-300/40 hover:text-stone-100"
                  onClick={interrupt}
                  disabled={!isConnected}
                >
                  Interrupt
                </button>
                <button
                  className="rounded-full border border-stone-500/30 px-5 py-3 text-sm font-semibold uppercase tracking-[0.35em] text-stone-200 transition hover:border-emerald-300/40 hover:text-stone-100"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={!isConnected}
                >
                  Send Image
                </button>
                <input
                  ref={fileInputRef}
                  className="hidden"
                  type="file"
                  accept="image/*"
                  onChange={handleFileInputChange}
                />
              </div>

              <div className="mt-4 rounded-2xl border border-stone-500/30 bg-stone-950/45 px-4 py-3">
                <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-stone-400">
                  <span>Accepted: JPG, PNG, WebP</span>
                  <span>Max file size: {MAX_IMAGE_UPLOAD_MB} MB</span>
                  <span className="hidden sm:inline">Shortcuts: M mute, I upload image, Esc interrupt</span>
                </div>
                {uploadStatus ? (
                  <div className="mt-3" aria-live="polite">
                    <div className="flex items-center justify-between text-xs text-stone-300">
                      <span>{uploadStatus}</span>
                      <span>{uploadProgress}%</span>
                    </div>
                    <div className="mt-1 h-1.5 w-full rounded-full bg-stone-800">
                      <div
                        className="h-1.5 rounded-full bg-emerald-300 transition-all duration-200"
                        style={{ width: `${Math.max(0, Math.min(uploadProgress, 100))}%` }}
                      />
                    </div>
                  </div>
                ) : null}
              </div>

              <div className="mt-6 grid gap-4 sm:grid-cols-5">
                <div className="sm:col-span-3">
                  <div className="flex items-center gap-2">
                    <label htmlFor="image-prompt" className="text-[0.6rem] font-semibold uppercase tracking-[0.35em] text-stone-500">
                      Image Prompt
                    </label>
                    <div className="group relative inline-flex items-center">
                      <button
                        type="button"
                        aria-label="What this prompt does"
                        aria-describedby="image-prompt-tooltip"
                        className="inline-flex h-5 w-5 items-center justify-center rounded-full border border-stone-500/40 text-[0.65rem] font-semibold text-stone-300 transition hover:border-stone-300 hover:text-stone-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-300/70"
                      >
                        i
                      </button>
                      <div
                        id="image-prompt-tooltip"
                        role="tooltip"
                        className="pointer-events-none absolute left-7 top-1/2 z-20 w-72 -translate-y-1/2 rounded-xl border border-stone-500/40 bg-stone-950/95 px-3 py-2 text-[0.72rem] leading-relaxed text-stone-200 opacity-0 shadow-[0_16px_50px_rgba(15,23,42,0.45)] transition group-hover:opacity-100 group-focus-within:opacity-100"
                      >
                        This prompt is sent together with your uploaded image so the AI can interpret the image using your instructions.
                      </div>
                    </div>
                  </div>
                  <input
                    id="image-prompt"
                    className="mt-2 w-full rounded-2xl border border-stone-500/30 bg-stone-950/45 px-4 py-3 text-sm text-stone-100 shadow-[0_12px_60px_rgba(15,23,42,0.4)] focus:border-emerald-300/60 focus:outline-none focus:ring-0"
                    value={promptText}
                    onChange={(event) => setPromptText(event.target.value)}
                    placeholder="Describe how the assistant should interpret the uploaded image"
                  />
                </div>
                <div className="flex flex-col justify-end gap-3 sm:col-span-2">
                  <div className="rounded-2xl border border-stone-500/30 bg-stone-950/45 px-4 py-3 text-[0.65rem] uppercase tracking-[0.35em] text-stone-400">
                    <span className="flex items-center justify-between text-stone-300">
                      <span>Capture</span>
                      <span className={`font-semibold ${isMicLive ? 'text-emerald-200' : isMuted ? 'text-stone-500' : 'text-stone-300'}`}>
                        {isConnected ? (isMicLive ? 'Streaming' : isMuted ? 'Muted' : 'Idle') : 'Offline'}
                      </span>
                    </span>
                  </div>
                </div>
              </div>

              <div className="mt-6 grid gap-3 sm:grid-cols-2">
                <div className="rounded-2xl border border-stone-500/30 bg-stone-950/45 px-4 py-3">
                  <span className="text-[0.55rem] font-semibold uppercase tracking-[0.4em] text-stone-500">Session</span>
                  <span className="mt-2 block truncate text-sm text-stone-200" suppressHydrationWarning>
                    {sessionId || '-'}
                  </span>
                </div>
                <div className="rounded-2xl border border-stone-500/30 bg-stone-950/45 px-4 py-3">
                  <span className="text-[0.55rem] font-semibold uppercase tracking-[0.4em] text-stone-500">Realtime Endpoint</span>
                  <span className="mt-2 block truncate text-sm text-stone-200" suppressHydrationWarning>
                    {sessionId ? buildWsUrl(wsBase, sessionId, memoryKey) : `${wsBase}/ws/{pending}`}
                  </span>
                </div>
              </div>

              {sessionSummary ? (
                <div className="mt-4 rounded-2xl border border-stone-500/30 bg-stone-950/45 px-4 py-4">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="text-[0.55rem] font-semibold uppercase tracking-[0.4em] text-stone-500">Session Summary</span>
                    <span className="text-xs text-stone-400">{new Date(sessionSummary.endedAt).toLocaleTimeString()}</span>
                  </div>
                  <div className="mt-3 grid gap-2 sm:grid-cols-3">
                    <div className="rounded-xl border border-stone-500/25 bg-stone-900/35 px-3 py-2 text-xs text-stone-300">
                      User turns: <span className="font-semibold text-stone-100">{sessionSummary.userTurns}</span>
                    </div>
                    <div className="rounded-xl border border-stone-500/25 bg-stone-900/35 px-3 py-2 text-xs text-stone-300">
                      Assistant turns: <span className="font-semibold text-stone-100">{sessionSummary.assistantTurns}</span>
                    </div>
                    <div className="rounded-xl border border-stone-500/25 bg-stone-900/35 px-3 py-2 text-xs text-stone-300">
                      Tool calls: <span className="font-semibold text-stone-100">{sessionSummary.toolCalls}</span>
                    </div>
                  </div>
                  <div className="mt-3 grid gap-2 sm:grid-cols-2">
                    <div className="rounded-xl border border-stone-500/25 bg-stone-900/35 px-3 py-2 text-xs text-stone-300">
                      Response mode: <span className="font-semibold text-stone-100">{sessionSummary.responseMode}</span>
                    </div>
                    <div className="rounded-xl border border-stone-500/25 bg-stone-900/35 px-3 py-2 text-xs text-stone-300 truncate">
                      Memory key: <span className="font-semibold text-stone-100">{sessionSummary.memoryKey}</span>
                    </div>
                  </div>
                  <p className="mt-3 rounded-xl border border-stone-500/25 bg-stone-900/35 px-3 py-2 text-xs leading-relaxed text-stone-300">
                    <span className="font-semibold text-stone-100">Last assistant reply:</span>{' '}
                    {sessionSummary.lastAssistantReply}
                  </p>
                </div>
              ) : null}
            </div>

            {/* Right Toggle Button - Feed */}
            <button
              onClick={() => setRightPanelOpen(!rightPanelOpen)}
              className="group flex h-20 w-24 flex-col items-center justify-center self-start rounded-2xl border border-stone-500/30 bg-stone-900/40 backdrop-blur-2xl transition-all hover:border-stone-400/60 hover:bg-stone-900/55 sm:-translate-y-2"
            >
              <div className="mb-2 flex flex-col items-center gap-1">
                <svg className={`h-3 w-3 text-stone-400 transition-all group-hover:text-stone-100 ${rightPanelOpen ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
                <svg className="h-4 w-4 text-stone-300 transition-colors group-hover:text-stone-100" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 4V2a1 1 0 011-1h8a1 1 0 011 1v2m0 0V3a1 1 0 011 1v11a1 1 0 01-1 1H8a1 1 0 01-1-1V4m0 0H5a1 1 0 00-1 1v11a1 1 0 001 1h1m4-10h2m0 0V4m0 2v2m0-2h2" />
                </svg>
              </div>
              <span className="text-[0.6rem] font-semibold uppercase tracking-[0.3em] text-stone-300 transition-colors group-hover:text-stone-100">
                Feed
              </span>
            </button>
          </div>
        </section>

      </main>
    </div>
  );
}
