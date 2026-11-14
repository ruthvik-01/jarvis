using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Windows.Forms;

namespace Jarvis
{
    internal static class Program
    {
        [STAThread]
        private static int Main(string[] arguments)
        {
            try
            {
                string executableDirectory = AppDomain.CurrentDomain.BaseDirectory;
                string directEntryScript = Path.Combine(executableDirectory, "jarvis.py");
                string nestedEntryScript = Path.Combine(executableDirectory, "windows", "jarvis.py");
                string entryScriptPath = File.Exists(directEntryScript) ? directEntryScript : nestedEntryScript;

                if (!File.Exists(entryScriptPath))
                {
                    MessageBox.Show(
                        "Could not locate jarvis.py next to Jarvis.exe or inside windows\\jarvis.py.",
                        "Jarvis Error",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Error
                    );
                    return 1;
                }

                string scriptDirectory = Path.GetDirectoryName(entryScriptPath) ?? executableDirectory;
                bool isConsoleMode = arguments != null && arguments.Length > 0;
                string pythonExecutable = FindPythonExecutable(preferWindowless: !isConsoleMode);
                if (string.IsNullOrEmpty(pythonExecutable))
                {
                    MessageBox.Show(
                        "Could not find a Python 3 installation (pythonw.exe / python.exe) on PATH or in AppData.",
                        "Jarvis Error",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Error
                    );
                    return 1;
                }

                List<string> argumentTokens = new List<string>();
                argumentTokens.Add(QuoteArgument(entryScriptPath));
                if (arguments != null)
                {
                    foreach (string singleArg in arguments)
                    {
                        argumentTokens.Add(QuoteArgument(singleArg));
                    }
                }

                ProcessStartInfo startInfo = new ProcessStartInfo();
                startInfo.FileName = pythonExecutable;
                startInfo.Arguments = string.Join(" ", argumentTokens.ToArray());
                startInfo.WorkingDirectory = scriptDirectory;
                startInfo.UseShellExecute = false;
                startInfo.CreateNoWindow = true;

                if (isConsoleMode)
                {
                    startInfo.RedirectStandardOutput = true;
                    startInfo.RedirectStandardError = true;
                    startInfo.StandardOutputEncoding = Encoding.UTF8;
                    startInfo.StandardErrorEncoding = Encoding.UTF8;

                    using (Process childProcess = new Process())
                    {
                        childProcess.StartInfo = startInfo;
                        childProcess.OutputDataReceived += delegate(object sender, DataReceivedEventArgs eventArgs)
                        {
                            if (eventArgs.Data != null)
                            {
                                Console.Out.WriteLine(eventArgs.Data);
                            }
                        };
                        childProcess.ErrorDataReceived += delegate(object sender, DataReceivedEventArgs eventArgs)
                        {
                            if (eventArgs.Data != null)
                            {
                                Console.Error.WriteLine(eventArgs.Data);
                            }
                        };
                        childProcess.Start();
                        childProcess.BeginOutputReadLine();
                        childProcess.BeginErrorReadLine();
                        childProcess.WaitForExit();
                        return childProcess.ExitCode;
                    }
                }

                using (Process guiProcess = Process.Start(startInfo))
                {
                    return guiProcess != null ? 0 : 1;
                }
            }
            catch (Exception exception)
            {
                MessageBox.Show(
                    "Failed to start Jarvis:\n\n" + exception.Message,
                    "Jarvis Error",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
                return 1;
            }
        }

        private static string QuoteArgument(string rawArgument)
        {
            if (string.IsNullOrEmpty(rawArgument))
            {
                return "\"\"";
            }
            if (rawArgument.IndexOfAny(new char[] { ' ', '\t', '"' }) < 0)
            {
                return rawArgument;
            }
            return "\"" + rawArgument.Replace("\"", "\\\"") + "\"";
        }

        private static string FindPythonExecutable(bool preferWindowless)
        {
            string exeName = preferWindowless ? "pythonw.exe" : "python.exe";
            string fallbackName = "python.exe";

            string pathVariable = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
            foreach (string pathSegment in pathVariable.Split(Path.PathSeparator))
            {
                string cleanSegment = pathSegment.Trim().Trim('"');
                if (string.IsNullOrEmpty(cleanSegment))
                {
                    continue;
                }
                string candidatePath = Path.Combine(cleanSegment, exeName);
                if (File.Exists(candidatePath) && candidatePath.IndexOf("WindowsApps", StringComparison.OrdinalIgnoreCase) < 0)
                {
                    return candidatePath;
                }
            }

            string localProgramsPython = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "Programs",
                "Python"
            );
            if (Directory.Exists(localProgramsPython))
            {
                string[] directories = Directory.GetDirectories(localProgramsPython, "Python3*");
                Array.Sort(directories, StringComparer.OrdinalIgnoreCase);
                Array.Reverse(directories);
                foreach (string pythonFolder in directories)
                {
                    string candidatePath = Path.Combine(pythonFolder, exeName);
                    if (File.Exists(candidatePath))
                    {
                        return candidatePath;
                    }
                    string fallbackCandidate = Path.Combine(pythonFolder, fallbackName);
                    if (File.Exists(fallbackCandidate))
                    {
                        return fallbackCandidate;
                    }
                }
            }

            return exeName;
        }
    }
}
