import { test, expect } from '@playwright/test';

test('video intensity analysis', async ({ page }) => {
  // Navigate to the app
  await page.goto('/');

  // Wait for the page to load
  await page.waitForLoadState('networkidle');

  // Start camera
  await page.click('button:has-text("Start Camera")');
  
  // Wait for video element to be visible
  await page.waitForSelector('video', { timeout: 5000 });

  // Connect to backend
  await page.click('button:has-text("Connect")');

  // Wait for connection to establish
  await page.waitForSelector('text=Connected', { timeout: 10000 });

  // Wait for overlay to show intensity data
  await page.waitForSelector('text=Intensity:', { timeout: 5000 });

  // Get intensity value from overlay
  const intensityText = await page.textContent('text=Intensity:');
  expect(intensityText).toMatch(/Intensity: \d+\.\d+ \/ 255/);

  // Extract intensity value (should be around 127.5 for gray50)
  const intensityMatch = intensityText?.match(/Intensity: (\d+\.\d+) \/ 255/);
  expect(intensityMatch).toBeTruthy();
  
  const intensity = parseFloat(intensityMatch![1]);
  // Gray50 should be around 127.5, allow ±5% tolerance
  expect(intensity).toBeGreaterThan(120);
  expect(intensity).toBeLessThan(135);

  // Check that messages per second is reasonable (should be around 30)
  const messagesText = await page.textContent('text=Messages/sec:');
  expect(messagesText).toMatch(/Messages\/sec: \d+\.\d+/);
  
  const messagesMatch = messagesText?.match(/Messages\/sec: (\d+\.\d+)/);
  expect(messagesMatch).toBeTruthy();
  
  const messagesPerSecond = parseFloat(messagesMatch![1]);
  expect(messagesPerSecond).toBeGreaterThan(28); // Should be at least 28 messages/sec

  // Wait a bit to ensure sustained performance
  await page.waitForTimeout(3000);

  // Check that metrics are still updating
  const updatedIntensityText = await page.textContent('text=Intensity:');
  expect(updatedIntensityText).not.toBe(intensityText); // Should have updated
});
