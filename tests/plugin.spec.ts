import { afterEach, describe, expect, it, vi } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import WorkerThreadCodeRuntime from '@deepseek-ai/dsh-code-runtime-worker-thread'
import { ToolCallId } from '@deepseek-ai/dsh-llm'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import ToolRuntime from '@deepseek-ai/dsh-tools'
import * as plugin from '../src/index.ts'
import { AutoToyBackend } from '../src/auto.ts'
import type { ToyDevice } from '../src/types.ts'

const toolNames = [
  'toy_connect', 'toy_control', 'toy_disconnect', 'toy_list',
  'toy_scan', 'toy_scan_raw_ble', 'toy_stop',
]
const devices: ToyDevice[] = [{
  id: 'buttplug:0',
  name: 'Fixture',
  features: [{ id: 'buttplug:0:0:vibrate', kind: 'vibrate', description: 'motor' }],
}]
const contexts = new Set<Context>()

afterEach(async () => {
  try {
    for (const ctx of contexts) await ctx.fiber.dispose()
  } finally {
    contexts.clear()
    vi.restoreAllMocks()
  }
})

async function mount(mode: 'native' | 'ptc' = 'native', config: plugin.Config = {}) {
  const ctx = new Context()
  contexts.add(ctx)
  await ctx.plugin(SystemPrompt)
  await ctx.plugin(ToolRuntime, { mode })
  if (mode === 'ptc') await ctx.plugin(WorkerThreadCodeRuntime)
  const fiber = ctx.plugin(plugin, { intifaceAutoDownload: false, ...config })
  await fiber
  return { ctx, fiber }
}

function mockBackend() {
  return {
    connect: vi.spyOn(AutoToyBackend.prototype, 'connect').mockResolvedValue({
      provider: 'buttplug', serverName: 'fixture', devices,
    }),
    scan: vi.spyOn(AutoToyBackend.prototype, 'scan').mockResolvedValue(devices),
    list: vi.spyOn(AutoToyBackend.prototype, 'list').mockReturnValue(devices),
    setLevel: vi.spyOn(AutoToyBackend.prototype, 'setLevel').mockResolvedValue(),
    stop: vi.spyOn(AutoToyBackend.prototype, 'stop').mockResolvedValue(),
    close: vi.spyOn(AutoToyBackend.prototype, 'close').mockResolvedValue(),
  }
}

function execute(ctx: Context, name: string, args: unknown = {}, signal = new AbortController().signal) {
  return ctx.tools.execute({ callId: ToolCallId(`test-${name}`), name, arguments: args, signal })
}

describe('DSH plugin integration', () => {
  it('provides a read-only host safety seam without exposing actuation or credentials', async () => {
    const backend = mockBackend()
    const { ctx } = await mount('ptc', { maxIntensityPercent: 40, maxDurationSeconds: 30 })
    const signal = new AbortController().signal
    expect(Object.keys(ctx.toySafety).sort()).toEqual(['devices', 'limits', 'stop'])
    expect(ctx.toySafety.limits).toEqual({ maxIntensityPercent: 40, maxDurationSeconds: 30 })
    const snapshot = await ctx.toySafety.devices(signal)
    snapshot[0]!.name = 'mutated'
    expect((await ctx.toySafety.devices(signal))[0]!.name).toBe('Fixture')
    await ctx.toySafety.stop(signal)
    expect(backend.stop).toHaveBeenCalledWith(undefined, signal)
    expect(backend.setLevel).not.toHaveBeenCalled()
  })

  it('registers all tools and publishes their schemas without configuration secrets', async () => {
    const token = 'fixture-private-sharing-token'
    const { ctx } = await mount('native', { monsterPartySessionToken: token })
    expect(ctx.tools.schemas().map(tool => tool.name).sort()).toEqual(toolNames)
    const assembly = await ctx.systemPrompt.assemble()
    expect(assembly.tools.map(tool => tool.name).sort()).toEqual(toolNames)
    expect(JSON.stringify(assembly)).not.toContain(token)
    expect(await execute(ctx, 'toy_list')).toEqual({
      isError: false, value: [], content: [{ type: 'text', text: '[]' }],
    })
  })

  it('executes the connection lifecycle through the native registry with canonical JSON results', async () => {
    const backend = mockBackend()
    const { ctx } = await mount()
    const signal = new AbortController().signal
    const connection = await execute(ctx, 'toy_connect', { model: 'Fixture' }, signal)
    expect(connection).toMatchObject({ isError: false, value: { serverName: 'fixture', devices } })
    expect(backend.connect).toHaveBeenCalledWith(signal, { model: 'Fixture' })
    for (const name of ['toy_scan', 'toy_list']) {
      expect(await execute(ctx, name)).toEqual({
        isError: false, value: devices, content: [{ type: 'text', text: JSON.stringify(devices) }],
      })
    }
    const value = { deviceId: 'buttplug:0', kind: 'vibrate', intensityPercent: 25, autoStopSeconds: 30 }
    expect(await execute(ctx, 'toy_control', {
      device_id: 'buttplug:0', kind: 'vibrate', intensity_percent: 25,
    })).toEqual({ isError: false, value, content: [{ type: 'text', text: JSON.stringify(value) }] })
    expect(await execute(ctx, 'toy_stop')).toMatchObject({ isError: false, value: { stopped: 'all' } })
    expect(backend.stop).toHaveBeenCalledWith(undefined, expect.any(AbortSignal))
    expect(await execute(ctx, 'toy_disconnect')).toMatchObject({ isError: false, value: { disconnected: true } })
    expect(backend.close).toHaveBeenCalledOnce()
  })

  it('rejects malformed arguments and safety violations before sending commands', async () => {
    const backend = mockBackend()
    const { ctx } = await mount('native', {
      defaultDurationSeconds: 1, maxDurationSeconds: 5, maxIntensityPercent: 60,
    })
    expect((await execute(ctx, 'toy_connect', {})).isError).toBe(true)
    expect(backend.connect).not.toHaveBeenCalled()
    for (const overrides of [
      { intensity_percent: '25' },
      { intensity_percent: 61 },
      { intensity_percent: 0.5 },
      { duration_seconds: 0 },
      { duration_seconds: 6 },
      { kind: 'unsupported' },
    ]) {
      const result = await execute(ctx, 'toy_control', {
        device_id: 'buttplug:0', kind: 'vibrate', intensity_percent: 25, ...overrides,
      })
      expect(result.isError).toBe(true)
    }
    expect(backend.setLevel).not.toHaveBeenCalled()
  })

  it('supports canonical tool returns through the published PTC worker runtime', async () => {
    const backend = mockBackend()
    const { ctx } = await mount('ptc')
    const assembly = await ctx.systemPrompt.assemble()
    expect(assembly.tools.map(tool => tool.name)).toEqual(['run_code'])
    expect(assembly.sections.find(section => section.name === 'tools:sdk')?.text).toContain('toy_control')
    const result = await execute(ctx, 'run_code', {
      description: 'Check the toy plugin with fixture devices',
      code: `
        const devices = await tools.toy_list({});
        const controlled = await tools.toy_control({
          device_id: devices[0].id, kind: 'vibrate', intensity_percent: 0
        });
        console.log(JSON.stringify({ devices, controlled }));
      `,
    })
    expect(result.isError).toBe(false)
    const text = result.content.flatMap(block => block.type === 'text' ? [block.text] : []).join('\n')
    expect(text).toContain(JSON.stringify({
      devices,
      controlled: { deviceId: 'buttplug:0', kind: 'vibrate', intensityPercent: 0, autoStopSeconds: null },
    }))
    expect(backend.setLevel).toHaveBeenCalledOnce()
  })

  it('does not dispatch a call whose signal is already aborted', async () => {
    const backend = mockBackend()
    const { ctx } = await mount()
    const controller = new AbortController()
    controller.abort()
    expect((await execute(ctx, 'toy_list', {}, controller.signal)).isError).toBe(true)
    expect(backend.list).not.toHaveBeenCalled()
  })

  it('awaits transport shutdown, unregisters tools, and allows a fresh plugin load', async () => {
    const backend = mockBackend()
    const { ctx, fiber } = await mount()
    let release!: () => void
    backend.close.mockImplementationOnce(() => new Promise<void>(resolve => { release = resolve }))
    let disposed = false
    const disposing = fiber.dispose().then(() => { disposed = true })
    try {
      await vi.waitFor(() => expect(backend.close).toHaveBeenCalledOnce())
      expect(disposed).toBe(false)
    } finally {
      release()
      await disposing
    }
    expect(ctx.tools.schemas()).toEqual([])
    expect((await execute(ctx, 'toy_list')).isError).toBe(true)
    await ctx.plugin(plugin, { intifaceAutoDownload: false })
    expect(ctx.tools.schemas().map(tool => tool.name).sort()).toEqual(toolNames)
  })
})
